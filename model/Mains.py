
import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
from model.teacher import STMLP as Network
from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
from lib.metrics import MAE_torch
from lib.pre_graph import get_adjacency_matrix

#*************************************************************************#
Mode = 'train'
# Mode = 'test'
DEBUG = True            #Save model parameters or not
DATASET = 'PEMS04'      #PEMSD4, PEMSD8, PEMSD3, PEMSD7
DEVICE = 'cuda:0'
MODEL = 'MTGNN'


#get configuration
config_file = './{}_{}.conf'.format(DATASET, MODEL)
#print('Read configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)


def scaler_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

def Mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)

#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument('--mode', default= Mode, type=str)
args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--model', default=MODEL, type=str)
args.add_argument('--cuda', default=True, type=bool)
#data
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--lag', default=config['data']['lag'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--tod', default=config['data']['tod'], type=eval)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
args.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
#model
args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument('--input_window', default=config['model']['input_window'], type=int)
args.add_argument('--output_window', default=config['model']['output_window'], type=int)
# model_mtgmm
args.add_argument('--gcn_true', type=eval, default=config['model']['gcn_true'])
args.add_argument('--buildA_true', type=eval, default=config['model']['buildA_true'])
args.add_argument('--gcn_depth', type=int, default=config['model']['gcn_depth'])
args.add_argument('--dropout', type=float, default=config['model']['dropout'])
args.add_argument('--subgraph_size', type=int, default=config['model']['subgraph_size'])
args.add_argument('--node_dim', type=int, default=config['model']['node_dim'])
args.add_argument('--dilation_exponential', type=int, default=config['model']['dilation_exponential'])
args.add_argument('--conv_channels', type=int, default=config['model']['conv_channels'])
args.add_argument('--residual_channels', type=int, default=config['model']['residual_channels'])
args.add_argument('--skip_channels', type=int, default=config['model']['skip_channels'])
args.add_argument('--end_channels', type=int, default=config['model']['end_channels'])
args.add_argument('--layers', type=int, default=config['model']['layers'])
args.add_argument('--propalpha', type=float, default=config['model']['propalpha'])
args.add_argument('--tanhalpha', type=int, default=config['model']['tanhalpha'])
args.add_argument('--layer_norm_affline', type=eval, default=config['model']['layer_norm_affline'])
args.add_argument('--use_curriculum_learning', type=eval, default=config['model']['use_curriculum_learning'])
args.add_argument('--task_level', type=int, default=config['model']['task_level'])
#train
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--teacher_forcing', default=False, type=bool)
#args.add_argument('--tf_decay_steps', default=2000, type=int, help='teacher forcing decay steps')
args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
#test
args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
#log
args.add_argument('--log_dir', default='./', type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)
args.add_argument('--teacher', default=True, type=eval)
args = args.parse_args()

args.filepath = '../PEMS_data/' + DATASET +'/'
if DATASET == 'PEMS03':
    A, Distance = get_adjacency_matrix(distance_df_filename=args.filepath + DATASET + '.csv', num_of_vertices=args.num_nodes, id_filename=args.filepath + DATASET + '.txt')
else:
    A, Distance = get_adjacency_matrix(distance_df_filename=args.filepath + DATASET + '.csv', num_of_vertices=args.num_nodes)
args.adj_mx = A

init_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'

#init model
model = Network(args)
model = model.to(args.device)
# for p in model.parameters():
#     if p.dim() > 1:
#         nn.init.xavier_uniform_(p)
#     else:
#         nn.init.uniform_(p)
print_model_parameters(model, only_num=False)

#load dataset
train_loader, val_loader, test_loader, scaler_data, scaler_day, scaler_week = get_dataloader(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, dow=False,
                                                               weather=False, single=False)

#init loss function, optimizer
if args.loss_func == 'mask_mae':
    loss = scaler_mae_loss(scaler_data, mask_value=0.0)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
else:
    raise ValueError

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=0, amsgrad=False)
#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)

#config log path
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir,'SAVE', args.dataset)
Mkdir(log_dir)
args.log_dir = log_dir

#start training
trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler_data,
                  args, lr_scheduler=lr_scheduler)
if args.mode == 'train':
    trainer.trainS()
elif args.mode == 'test':
    model.load_state_dict(torch.load(log_dir + '/best_modelstudent.pth'))
    print("Load saved model")
    trainer.test(model, trainer.args, test_loader, scaler_data, trainer.logger)
else:
    raise ValueError
