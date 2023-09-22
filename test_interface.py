"""
Test interface for speech enhancement!
You can just run this file.
"""
import os
import argparse
import torch
import options
import datetime
import random
import utils
import numpy as np
from dataloader import create_dataloader_test
import pandas as pd

######################################################################################################################
#                                                  Parser init                                                       #
######################################################################################################################
opt = options.Options().init(argparse.ArgumentParser(description='speech enhancement')).parse_args()
print(opt)

######################################################################################################################
#                                    Set a model (check point) and a log folder                                      #
######################################################################################################################
dir_name = os.path.dirname(os.path.abspath(__file__))  # absolute path
print(dir_name)

log_dir = os.path.join(dir_name, 'log', opt.arch + '_' + opt.env)

utils.mkdir(log_dir)
print("Now time is : ", datetime.datetime.now().isoformat())
tboard_dir = './log/{}_{}/logs'.format(opt.arch, opt.env)  # os.path.join(log_dir, 'logs')
model_dir = './log/{}_{}/models'.format(opt.arch, opt.env)  # os.path.join(log_dir, 'models')
utils.mkdir(model_dir)  # make a dir if there is no dir (given path)
utils.mkdir(tboard_dir)

######################################################################################################################
#                                                   Model init                                                       #
######################################################################################################################
# set device
DEVICE = torch.device(opt.device)

# set seeds
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# define model
model_adult, model_child = utils.get_arch_test(opt)

total_params = utils.cal_total_params(model_adult)
print('total params   : %d (%.2f M, %.2f MBytes)\n' %
      (total_params,
       total_params / 1000000.0,
       total_params * 4.0 / 1000000.0))

# load the params
print('Load the pretrained model...')
chkpt = torch.load(opt.weights_adult)
model_adult.load_state_dict(chkpt['model'])

chkpt = torch.load(opt.weights_child)
model_child.load_state_dict(chkpt['model'])

model_adult = model_adult.to(DEVICE)
model_child = model_child.to(DEVICE)

######################################################################################################################
######################################################################################################################
#                                             Main program - test                                                    #
######################################################################################################################
######################################################################################################################
test_loader_adult, test_loader_child = create_dataloader_test(opt)
test_log_fp = open(model_dir + '/test_log.txt', 'a')

t_all = []
o_all = []
b_all = []

submission = pd.read_csv('../Dataset/MAIC2023/submission.csv')

model_adult.eval()
with torch.no_grad():
    for inputs, name in utils.Bar(test_loader_adult):
        # to cuda
        inputs = inputs.float().to(DEVICE)

        outputs = model_adult(inputs)
        outputs = 103 * torch.sigmoid(outputs) + 19  # adult

        age_output = outputs.item()
        submission.loc[submission['FILENAME'] == name[0], 'AGE'] = age_output

model_child.eval()
with torch.no_grad():
    for inputs, name in utils.Bar(test_loader_child):
        # to cuda
        inputs = inputs.float().to(DEVICE)

        outputs = model_child(inputs)
        outputs = 9 * torch.sigmoid(outputs)  # adult

        age_output = outputs.item()
        submission.loc[submission['FILENAME'] == name[0], 'AGE'] = age_output

submission.to_csv('./score/submission.csv', index=False)
