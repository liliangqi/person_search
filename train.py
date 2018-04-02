# -----------------------------------------------------
# Train Spatial Invariant Person Search Network
#
# Author: Liangqi Li
# Creating Date: Mar 31, 2018
# Latest rectified: Apr 2, 2018
# -----------------------------------------------------
import os
import sys
import argparse

import numpy as np
import torch
import yaml
import torch.nn as nn
from torch.autograd import Variable
import time

from dataset import PersonSearchDataset
from model import SIPN


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--net', default='res50', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--out_dir', default='./output', type=str)
    parser.add_argument('--pre_model', default='', type=str)

    args = parser.parse_args()

    return args


def cuda_mode(args):
    """set cuda"""
    if torch.cuda.is_available() and '-1' not in args.gpu_ids:
        cuda = True
        str_ids = args.gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            gid = int(str_id)
            if gid >= 0:
                gpu_ids.append(gid)

        if len(gpu_ids) > 0:
            torch.cuda.set_device(gpu_ids[0])
    else:
        cuda = False

    return cuda


def train_model(dataset, net, learning_rate, optimizer, num_epochs):
    """Train the model"""
    pass



def main():

    opt = parse_args()
    use_cuda = cuda_mode(opt)
    network = SIPN(opt.net, opt.pre_model)
    save_dir = opt.out_dir
    lr = opt.lr
    with open('config.yml', 'r') as f:
        config = yaml.load(f)

    params = []
    for key, value in dict(network.named_parameters()).items():
        if value.requires_grad:
            # TODO: set different decay for weight and bias
            params += [{'params': [value], 'lr': lr, 'weight_decay': 5e-4}]

    if opt.optimizer == 'SGD':
        optimiz = torch.optim.SGD(params, momentum=0.9)
    elif opt.optimizer == 'Adam':
        lr *= 0.1
        optimiz = torch.optim.Adam(params)

    # TODO: add resume
    all_epoch_loss = 0

    for epoch in opt.epochs:
        network.train()
        epoch_start = time.time()
        start = epoch_start

        if use_cuda:
            network.cuda()
        if epoch > 0 and epoch % 2 == 0:
            lr *= config['gamma']  # TODO: use lr_scheduel
        # load data for each epoch
        dataset = PersonSearchDataset(opt.data_dir)

        for step in range(len(dataset)):
            im, gt_boxes, im_info = dataset.next()
            im = im.transpose([0, 3, 1, 2])

            if use_cuda:
                im = Variable(torch.from_numpy(im).cuda())
                gt_boxes = Variable(torch.from_numpy(gt_boxes).float().cuda())
            else:
                im = Variable(torch.from_numpy(im))
                gt_boxes = Variable(torch.from_numpy(gt_boxes).float())

            losses = network(im, gt_boxes, im_info)
            optimiz.zero_grad()
            losses.sum().backward()
            optimiz.step()

            for loss in losses:
                all_epoch_loss += loss.data
            current_iter = epoch * len(dataset) + step + 1
            average_loss = all_epoch_loss / current_iter

            if step % config['disp_interval'] == 0:
                end = time.time()
                print('Epoch {:2d}, iter {:5d}, average loss: {:.6f}, lr: '
                      '{:.2e}'.format(epoch, step, average_loss, lr))
                print('>>>> rpn_cls: {:.6f}'.format(losses[0].data))
                print('>>>> rpn_box: {:.6f}'.format(losses[1].data))
                print('>>>> cls: {:.6f}'.format(losses[2].data))
                print('>>>> box: {:.6f}'.format(losses[3].data))
                print('>>>> reid: {:.6f}'.format(losses[4].data))
                print('time cost: {:.3f}s'.format(end - start))

        epoch_end = time.time()
        print('\nEntire epoch time cost: {.2f} hours\n'.format(
            (epoch_end - epoch_start) / 3600))

        save_name = os.path.join(save_dir, 'sipn_{}.pth'.format(epoch))
        torch.save(network.state_dict(), save_name)












