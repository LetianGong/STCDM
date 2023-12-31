import os
import warnings
from absl import app, flags
import torch
import logging
import numpy as np
import pandas as pd
import argparse
import co_evolving_condition
from utils import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
warnings.filterwarnings("ignore", category=DeprecationWarning)

randomSeed = 2022
torch.manual_seed(randomSeed)
torch.cuda.manual_seed(randomSeed)
torch.cuda.manual_seed_all(randomSeed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(randomSeed)

FLAGS = flags.FLAGS
flags.DEFINE_string('dataroot', './STCDM_data/', help='dataset root')
flags.DEFINE_string('logdir', './STCDM_exp', help='log directory')
flags.DEFINE_bool('train', True, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate')
flags.DEFINE_bool('use_nni', False, help='if use nni')

# Network Architecture
flags.DEFINE_multi_integer('encoder_dim', None, help='encoder_dim')
flags.DEFINE_string('encoder_dim_con', "64,128,256", help='encoder_dim_con')
flags.DEFINE_integer('embed_dim', 512, help='nf')
flags.DEFINE_integer('input_size', None, help='input_size')
flags.DEFINE_integer('cond_size', None, help='cond_size')
flags.DEFINE_integer('output_size', None, help='output_size')
flags.DEFINE_string('activation', 'relu', help='activation')
flags.DEFINE_integer('POI_num', 1097, help='POI_num')
flags.DEFINE_integer('time_slot', 24, help='time_slot')

# Training
flags.DEFINE_integer('training_batch_size', 64, help='batch size')
flags.DEFINE_integer('eval_batch_size', 64, help='batch size')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_float('beta_1', 0.00001, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_float('lr_con', 2e-03, help='target learning rate')
flags.DEFINE_float('lr_dis', 2e-03, help='target learning rate')
flags.DEFINE_integer('total_epochs_both', 100, help='total training steps')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
flags.DEFINE_string('ctx', '0', help='ctx')

# Sampling
flags.DEFINE_integer('sample_step', 1, help='frequency of sampling')

# Continuous diffusion model
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedsmall', ['fixedlarge', 'fixedsmall'], help='variance type')

# Contrastive Learning
flags.DEFINE_integer('ns_method', 0, help='negative condition method')
flags.DEFINE_float('lambda_con', 0.2, help='lambda_con')
flags.DEFINE_float('lambda_dis', 0.2, help='lambda_dis')

#Dataset
flags.DEFINE_string('dataset', 'Weeplaces', help='dataset')


def main(argv):

    if FLAGS.eval == True:
        warnings.simplefilter(action='ignore', category=FutureWarning)
        os.makedirs(FLAGS.logdir,exist_ok=True)
        gfile_stream = open(os.path.join(FLAGS.logdir, 'eval.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
    else:
        warnings.simplefilter(action='ignore', category=FutureWarning)
        os.makedirs(FLAGS.logdir,exist_ok=True)
        gfile_stream = open(os.path.join(FLAGS.logdir, 'train.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')


    logging.info("Co-evolving Conditional Diffusion models")
    device = torch.device("cuda:"+FLAGS.ctx if torch.cuda.is_available() else "cpu")
    co_evolving_condition.train_ST(FLAGS)

if __name__ == '__main__':
    app.run(main)
