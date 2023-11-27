"""Return training and evaluation/test datasets from config files."""
import torch
import numpy as np
import json
import logging
import os
import numpy as np

def data_loader(FLAGS):

    if FLAGS.dataset == 'Caberra':
        f = open(FLAGS.dataroot + FLAGS.dataset + '-UserTraceData.npz','rb')
        train_data = np.load(f)
        train_data_mask = np.array([[0 if k < 0 else 1 for k in i]for i in train_data['TIME']])
        train_data = np.concatenate((train_data['SPACE'].reshape((train_data['SPACE'].shape[0],1,FLAGS.POI_num)), np.array([[-1 if k < 1 else np.log10(k) for k in i]for i in train_data['TIME']]).reshape((train_data['SPACE'].shape[0],1,FLAGS.POI_num)), train_data_mask.reshape((train_data['SPACE'].shape[0],1,FLAGS.POI_num))), axis = 1)
    else:
        f = open(FLAGS.dataroot + FLAGS.dataset + '-UserTraceData.npz','rb')
        train_data = np.load(f)
        train_data_mask = np.array([[0 if k < 0 else 1 for k in i]for i in train_data['TIME']])
        train_data = np.concatenate((train_data['SPACE'].reshape((train_data['SPACE'].shape[0],1,FLAGS.POI_num)), np.array([[-1 if k < 1 else k for k in i]for i in train_data['TIME']]).reshape((train_data['SPACE'].shape[0],1,FLAGS.POI_num)), train_data_mask.reshape((train_data['SPACE'].shape[0],1,FLAGS.POI_num))), axis = 1)

    data_all = train_data
    
    Max, Min, Mean = [], [], []
    for m in range(2):
        if m == 0:
            Max.append(np.max(data_all[:,m,:]))
            Min.append(np.min(data_all[:,m,:]))
            Mean.append(np.mean(data_all[:,m,:]))
        elif m==1:
            Max.append(max([max(i) for i in data_all[:,m,:].tolist()]))
            Min.append(0.)
            data_time = data_all.copy()
            data_time = data_time[:,m,:]
            data_time[data_time < 0] = 0
            Mean.append(np.mean(data_time))
    
    Max.append(1.)
    Min.append(0.)
    Mean.append(0)

    train_data = np.array([[[p if p<0. else ((p-Mean[k])/(Max[k]-Min[k])) for p in i] for k,i in enumerate(u)]for u in train_data]) 
    train_data[train_data < 0] = -1

    return train_data, Max, Min
