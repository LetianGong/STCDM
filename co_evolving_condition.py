import os
from absl import flags
import torch
import matplotlib.pyplot as plt
from diffusion_continuous import GaussianDiffusionTrainer, GaussianDiffusionSampler
import dataload
from torch.utils.data import DataLoader
from models.ST_unet import ST_Unet
import logging
import numpy as np
import pandas as pd
from utils import *
import nni

def train_ST(FLAGS):

    FLAGS = flags.FLAGS
    device = torch.device("cuda:"+FLAGS.ctx if torch.cuda.is_available() else "cpu")
    if FLAGS.dataset == 'Weeplaces':
        FLAGS.POI_num = 1097
    elif FLAGS.dataset == 'Gowalla':
        FLAGS.POI_num = 3028
    elif FLAGS.dataset == 'Dartmouth':
        FLAGS.POI_num = 576
    elif FLAGS.dataset == 'Caberra':
        FLAGS.POI_num = 317
    train_data, Max, Min = dataload.data_loader(FLAGS)
    FLAGS.input_size = train_data.shape[2]
    FLAGS.cond_size = 1
    FLAGS.output_size = train_data.shape[2]
    FLAGS.encoder_dim =  list(map(int, FLAGS.encoder_dim_con.split(',')))
    if FLAGS.use_nni:
        param = nni.get_next_parameter()
        FLAGS.T = int(param['T'])
        FLAGS.beta_1 = float(param['beta_1'])
        FLAGS.beta_T = float(param['beta_T'])
        FLAGS.lambda_con = float(param['lambda_con'])
        FLAGS.lambda_dis = float(param['lambda_dis'])
        FLAGS.embed_dim = int(param['embed_dim'])
    T = FLAGS.T
    beta_1 = FLAGS.beta_1
    beta_T = FLAGS.beta_T
    lambda_con = FLAGS.lambda_con
    lambda_dis = FLAGS.lambda_dis

    model_s = ST_Unet(FLAGS)
    optim_s = torch.optim.Adam(model_s.parameters(), lr=FLAGS.lr_con)
    sched_s = torch.optim.lr_scheduler.LambdaLR(optim_s, lr_lambda=warmup_lr) 
    trainer_s = GaussianDiffusionTrainer(model_s, beta_1, beta_T, T).to(device)
    net_sampler_s = GaussianDiffusionSampler(model_s, beta_1, beta_T, T, FLAGS.mean_type, FLAGS.var_type).to(device)

    model_t = ST_Unet(FLAGS)
    optim_t = torch.optim.Adam(model_t.parameters(), lr=FLAGS.lr_con)
    sched_t = torch.optim.lr_scheduler.LambdaLR(optim_t, lr_lambda=warmup_lr)
    trainer_t = GaussianDiffusionTrainer(model_t, beta_1, beta_T, T).to(device)
    net_sampler_t = GaussianDiffusionSampler(model_t, beta_1, beta_T, T, FLAGS.mean_type, FLAGS.var_type).to(device)

    if FLAGS.parallel:
        trainer_s = torch.nn.DataParallel(trainer_s)
        net_sampler_s = torch.nn.DataParallel(net_sampler_s)
        trainer_t = torch.nn.DataParallel(trainer_t)
        net_sampler_t = torch.nn.DataParallel(net_sampler_t)

    num_params_s = sum(p.numel() for p in model_s.parameters())
    num_params_t = sum(p.numel() for p in model_t.parameters())
    logging.info('Spatial model params: %d' % num_params_s)
    logging.info('Temporal model params: %d' % num_params_t)

    scores_max_eval = -10

    total_steps_both = FLAGS.total_epochs_both * int(train_data.shape[0]/FLAGS.training_batch_size+1)
    sample_step = FLAGS.sample_step * int(train_data.shape[0]/FLAGS.training_batch_size+1)

    logging.info("Total steps: %d" %total_steps_both)
    logging.info("Sample steps: %d" %sample_step)
    logging.info("Continuous: %d, %d" %(train_data.shape[0], train_data.shape[2]))

    if FLAGS.eval==False:
        epoch = 0
        train_iter_con = DataLoader(train_data, batch_size=FLAGS.training_batch_size)
        datalooper_train_con = infiniteloop(train_iter_con)

        p1 = np.zeros((FLAGS.time_slot,FLAGS.POI_num))
        q1 = np.zeros((FLAGS.time_slot,FLAGS.POI_num))
        p2 = np.zeros((FLAGS.POI_num))
        q2 = np.zeros((FLAGS.POI_num))
        p3 = np.zeros((FLAGS.POI_num,FLAGS.time_slot))
        q3 = np.zeros((FLAGS.POI_num,FLAGS.time_slot))

        x_all,sample_x_all = [],[]
        for step in range(total_steps_both):
            model_s.train()
            model_t.train()
            batch = next(datalooper_train_con).to(device).float()
            x_0_s, x_0_t, mask = batch2model(batch)
            ns_s, ns_t = ST_make_negative_condition(x_0_s, x_0_t)
            s_loss, s_loss_ns, t_loss, t_loss_ns = ST_training_with(x_0_s, x_0_t, trainer_s, trainer_t, ns_s, ns_t, FLAGS)

            loss_s = s_loss + lambda_con * s_loss_ns
            loss_t = t_loss + lambda_dis * t_loss_ns

            optim_s.zero_grad()
            loss_s.backward()
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), FLAGS.grad_clip)
            optim_s.step()
            sched_s.step()

            optim_t.zero_grad()
            loss_t.backward()
            torch.nn.utils.clip_grad_norm_(model_t.parameters(), FLAGS.grad_clip)
            optim_t.step()
            sched_t.step()

            if (step+1) % int(train_data.shape[0]/FLAGS.training_batch_size+1) == 0:
     
                logging.info(f"Epoch :{epoch}, diffusion spatial loss: {s_loss:.3f}, temporal loss: {t_loss:.3f}")
                logging.info(f"Epoch :{epoch}, CL spatial loss: {s_loss_ns:.3f}, temporal loss: {t_loss_ns:.3f}")
                logging.info(f"Epoch :{epoch}, Total spatial loss: {loss_s:.3f}, temporal loss: {loss_t:.3f}")
                epoch +=1
                if FLAGS.use_nni:
                    nni.report_intermediate_result(loss_s)

            if step > 0 and sample_step > 0 and step % sample_step == 0 or step==(total_steps_both-1):
                model_s.eval()
                model_t.eval()
                with torch.no_grad():
                    x_T_s = torch.randn(x_0_s.shape[0], x_0_s.shape[1]).to(device)
                    x_T_t = torch.randn(x_0_t.shape[0], x_0_t.shape[1]).to(device)
                    x_s, x_t = ST_sampling_with(x_T_s, x_T_t, net_sampler_s, net_sampler_t, FLAGS)
                    real_x,real_t = inverse_normalization(x_0_s.detach().cpu().numpy(), x_0_t.detach().cpu().numpy(), Max, Min)                   
                    sample_s,sample_t = inverse_normalization(x_s.detach().cpu().numpy(), x_t.detach().cpu().numpy(), Max, Min)
                    s,t = sample_wash(real_x,real_t,sample_s,sample_t)

            if step >= total_steps_both - int(train_data.shape[0]/FLAGS.training_batch_size+1):
                model_s.eval()
                model_t.eval()
                with torch.no_grad():
                    x_T_s = torch.randn(x_0_s.shape[0], x_0_s.shape[1]).to(device)
                    x_T_t = torch.randn(x_0_t.shape[0], x_0_t.shape[1]).to(device)
                    x_s, x_t = ST_sampling_with(x_T_s, x_T_t, net_sampler_s, net_sampler_t, FLAGS)
                    real_x,real_t = inverse_normalization(x_0_s.detach().cpu().numpy(), x_0_t.detach().cpu().numpy(), Max, Min)                   
                    sample_s,sample_t = inverse_normalization(x_s.detach().cpu().numpy(), x_t.detach().cpu().numpy(), Max, Min)
                    s,t = sample_wash(real_x,real_t,sample_s,sample_t)
                    for i in s:
                        sample_x_all.append(i[np.argmax(i)])
                    for j in real_x:
                        x_all.append(j[np.argmax(j)])
                    for k,i in enumerate(real_t):
                        for l,j in enumerate(i):
                            if j < 0:
                                continue 
                            for a in decode_numbers(int(j),int(real_x[k][l])):
                                p1[a][l] = p1[a][l] + 1
                    for k,i in enumerate(t):
                        for l,j in enumerate(i):
                            if isinstance(j, list):
                                for a in j:
                                    q1[a][l] = q1[a][l] + 1
                    for i in real_x:
                        p2[np.argmax(i)] = p2[np.argmax(i)] + 1
                    for i in s:
                        q2[np.argmax(i)] = q2[np.argmax(i)] + 1
                    for k,i in enumerate(real_t):
                        for l,j in enumerate(i):
                            if j < 0:
                                continue 
                            for a in decode_numbers(int(j),int(real_x[k][l])):
                                p3[l][a] = p3[l][a] + 1
                    for k,i in enumerate(t):
                        for l,j in enumerate(i):
                            if isinstance(j, list):
                                for a in j:
                                    q3[l][a] = q3[l][a] + 1
        js1 = 0
        js1_sum = 0
        for i in range(FLAGS.time_slot):
            js1 = js1 + js_divergence(p1[i],q1[i])*np.sum(p1[i])
            js1_sum = js1_sum + np.sum(p1[i])
        js2 = js_divergence(p2,q2)
        js3 = 0.
        js3_sum = 0.
        for i in range(FLAGS.POI_num):
            js3 = js3 + js_divergence(p3[i],q3[i]) * np.sum(p3[i])
            js3_sum = js3_sum + np.sum(p3[i])
        print('js_divergence:')
        print(js1/js1_sum,js2,js3/js3_sum)
        if FLAGS.use_nni:
            nni.report_final_result(js1/js1_sum+js2+js3/js3_sum)
        



