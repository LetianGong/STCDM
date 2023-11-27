import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.special import kl_div
from absl import flags


def warmup_lr(step):
    return min(step, 5000) / 5000

def infiniteloop(dataloader):
    while True:
        for _, y in enumerate(dataloader):
            yield y


def log_sample_categorical(logits, num_classes):
    full_sample = []
    k=0 
    for i in range(len(num_classes)):
        logits_column = logits[:,k:num_classes[i]+k]
        k+=num_classes[i]
        uniform = torch.rand_like(logits_column)
        gumbel_noise = -torch.log(-torch.log(uniform+1e-30)+1e-30) 
        sample = (gumbel_noise + logits_column).argmax(dim=1) 
        col_t = np.zeros(logits_column.shape) 
        col_t[np.arange(logits_column.shape[0]), sample.detach().cpu()] = 1 
        full_sample.append(col_t)
    full_sample = torch.tensor(np.concatenate(full_sample, axis=1))
    log_sample = torch.log(full_sample.float().clamp(min=1e-30))
    return log_sample


def add_dim_to_diffusionSeq(diffusion_seq, target_dim=3):
    if len(diffusion_seq.shape) < target_dim:
        return diffusion_seq.unsqueeze(-1)
    else:
        return diffusion_seq

def ST_sampling_with(x_T_s, x_T_t, net_sampler_s, net_sampler_t, FLAGS):

    x_t_s = x_T_s
    x_t_t = x_T_t
    T = FLAGS.T
    for time_step in reversed(range(T)):
        t = x_t_s.new_ones([x_t_s.shape[0], ], dtype=torch.long) * time_step
        #-----------------------------------------------------------------
        dynamic_cond_s = add_dim_to_diffusionSeq(x_t_t)
        cond_s = dynamic_cond_s 
        mean, log_var = net_sampler_s.p_mean_variance(x_t=x_t_s, t=t, cond = cond_s.to(x_t_s.device), trans=None) 
        if time_step > 0:
            noise = torch.randn_like(x_t_s)
        elif time_step == 0: 
            noise = 0
        x_t_minus_1_s = mean + torch.exp(0.5 * log_var) * noise
        x_t_minus_1_s = torch.clip(x_t_minus_1_s, -1., 1.)
        #-----------------------------------------------------------------
        dynamic_cond_t = add_dim_to_diffusionSeq(x_t_s)
        cond_t = dynamic_cond_t
        mean, log_var = net_sampler_t.p_mean_variance(x_t=x_t_t, t=t, cond = cond_t.to(x_t_t.device), trans=None)
        if time_step > 0:
            noise = torch.randn_like(x_t_t)
        elif time_step == 0:
            noise = 0
        x_t_minus_1_t = mean + torch.exp(0.5 * log_var) * noise
        x_t_minus_1_t = torch.clip(x_t_minus_1_t, -1., 1.) 
        #-----------------------------------------------------------------
        x_t_s = x_t_minus_1_s
        x_t_t = x_t_minus_1_t

    return x_t_s, x_t_t


def ST_training_with(x_0_s, x_0_t, trainer_s, trainer_t, ns_s, ns_t, FLAGS):
    T = FLAGS.T
    t = torch.randint(T, size=(x_0_s.shape[0], ), device=x_0_s.device)

    '''co-evolving training and predict positive samples'''
    noise_s = torch.randn_like(x_0_s)
    x_t_s = trainer_s.make_x_t(x_0_s, t, noise_s)
    noise_t = torch.randn_like(x_0_t)
    x_t_t = trainer_t.make_x_t(x_0_t, t, noise_t)
    #-----------------------------------------------------------------
    dynamic_cond_s = add_dim_to_diffusionSeq(x_t_t)
    cond_s = dynamic_cond_s
    eps_s = trainer_s.model(x_t_s, t, cond_s.to(x_t_s.device))
    ps_0_s = trainer_s.predict_xstart_from_eps(x_t_s, t, eps=eps_s)
    s_loss = F.mse_loss(eps_s, noise_s, reduction='none')
    s_loss = s_loss.mean()
    #-----------------------------------------------------------------
    dynamic_cond_t = add_dim_to_diffusionSeq(x_t_s)
    cond_t = dynamic_cond_t 
    eps_t = trainer_t.model(x_t_t, t, cond_t.to(x_t_t.device))
    ps_0_t = trainer_t.predict_xstart_from_eps(x_t_t, t, eps=eps_t)
    t_loss = F.mse_loss(eps_t, noise_t, reduction='none')
    t_loss = t_loss.mean()

    '''negative condition -> predict negative samples'''
    noise_ns_s = torch.randn_like(ns_s)
    ns_t_s = trainer_s.make_x_t(ns_s, t, noise_ns_s)
    noise_ns_t = torch.randn_like(ns_t) 
    ns_t_t = trainer_t.make_x_t(ns_t, t, noise_ns_t)
    #-----------------------------------------------------------------
    dynamic_cond_s = add_dim_to_diffusionSeq(ns_t_t)
    ns_cond_s = dynamic_cond_s
    eps_ns_s = trainer_s.model(x_t_s, t, ns_cond_s.to(x_t_s.device))
    ns_0_s = trainer_s.predict_xstart_from_eps(x_t_s, t, eps=eps_ns_s)

    dynamic_cond_t = add_dim_to_diffusionSeq(ns_t_s)
    ns_cond_t = dynamic_cond_t
    eps_ns_t = trainer_t.model(x_t_t, t, ns_cond_t.to(x_t_t.device))
    ns_0_t = trainer_t.predict_xstart_from_eps(x_t_t, t, eps=eps_ns_t)
    

    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    triplet_s = triplet_loss(x_0_s, ps_0_s, ns_0_s)
    #-----------------------------------------------------------------
    ps_t = F.cross_entropy(ps_0_t, torch.argmax(x_0_t, dim=-1).long(), reduction='none')
    ns_t = F.cross_entropy(ns_0_t, torch.argmax(x_0_t, dim=-1).long(), reduction='none')
    triplet_t = max((ps_t-ns_t).mean()+1, 0)

    return s_loss, triplet_s, t_loss, triplet_t


def ST_make_negative_condition(x_0_s, x_0_t):
    device = x_0_s.device
    x_0_s = x_0_s.detach().cpu().numpy()
    x_0_t = x_0_t.detach().cpu().numpy()

    ns_s_raw = pd.DataFrame(x_0_s)
    ns_t_raw = pd.DataFrame(x_0_t)
    x_0_s_len, x_0_t_len = x_0_s.shape[1], x_0_t.shape[1]

    nss_array = np.array(ns_s_raw.sample(frac=1, replace = False).reset_index(drop=True))
    nst_array = np.array(ns_t_raw.sample(frac=1, replace = False).reset_index(drop=True))
    ns_s = nss_array[:,:x_0_s_len].astype(np.float32)
    ns_t = nst_array[:,:x_0_t_len].astype(np.float32)

    return torch.tensor(ns_s).to(device), torch.tensor(ns_t).to(device)


def batch2model(batch, dim=2):

    device = batch.device

    if dim == 1:
        spatial = batch[:,0]
        mask = batch[:,-1,:]

    if dim == 2:
        spatial = batch[:,0]
        temporal = batch[:,1,:]
        mask = batch[:,-1,:]

    if dim == 3:
        spatial = batch[:,0]
        temporal = batch[:,1,:]
        mask = batch[:,-1,:]

    spatial = spatial.to(device)
    temporal = temporal.to(device)
    mask = mask.to(device)

    return spatial*mask, temporal, mask

def inverse_normalization(x_0_s, x_0_t, Max, Min):
    x_s = x_0_s * (Max[0] - Min[0])
    x_t = []
    for k,i in enumerate(x_0_t):
        i = i * (Max[1] - Min[1])
        x_t.append([])
        for u in i:
            if u > 0:
                x_t[k].append(np.power(10,u))
            else:
                x_t[k].append(-1.)
    x_t = (np.array(x_t))
    return  np.round(x_s), np.round(x_t)

def sample_wash(real_x,real_t,sample_s,sample_t):
    sample_s = np.where(sample_s<=0, 0, sample_s)
    sample_t = np.where(sample_s<=0, -1, sample_t)
    sample_s = np.where(sample_t<=0, 0, sample_s)
    sample_t = np.where(sample_t<=0, -1, sample_t)
    lenth = np.sum(real_x,axis=-1)
    s = []
    t = [] 
    for k,i in enumerate(sample_s):
        s_ = []
        t_ = []
        for l,j in enumerate(i):
            if j > 0:
                s_.append(j)
                t_.append(list(set(decode_numbers(int(sample_t[k][l]),int(j)))))
            else:
                s_.append(0)
                t_.append(-1)
        t.append(t_)
        s.append(s_)

    return s,t

def decode_numbers(encoded, n):
    binary_string = bin(encoded)[2:].zfill(5 * n)
    decoded_numbers = []
    for i in range(n):
        binary = binary_string[i * 5: (i + 1) * 5]
        number = int(binary, 2) + 1  
        decoded_numbers.append(number)
    decoded_numbers = [(i - 1)%24 for i in decoded_numbers]
    return decoded_numbers


def js_divergence(p,q):
    if np.sum(p) != 0:
        p = p / np.sum(p)
    else:
        p = 0
    if np.sum(q) != 0:
        q = q / np.sum(q)
    else:
        q = 0

    m = 0.5 * (p + q)
    kl_p = kl_div(p,m)
    kl_q = kl_div(q,m)
    
    js = 0.5 * (np.sum(kl_p) + np.sum(kl_q))

    return js 
