# -----------------------------------------------------
# Train Spatial Invariant Person Search Network
#
# Author: Liangqi Li
# Creating Date: Mar 31, 2018
# Latest rectified: Apr 9, 2018
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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    lr = opt.lr
    with open('config.yml', 'r') as f:
        config = yaml.load(f)

    params = []
    times1 = 0
    times2 = 0
    for key, value in dict(network.named_parameters()).items():
        times1 += 1
        if value.requires_grad:
            print(key)
            times2 += 1
            # TODO: set different decay for weight and bias
            params += [{'params': [value], 'lr': lr, 'weight_decay': 1e-4}]

    if opt.optimizer == 'SGD':
        optimiz = torch.optim.SGD(params, momentum=0.9)
    elif opt.optimizer == 'Adam':
        lr *= 0.1
        optimiz = torch.optim.Adam(params)

    # TODO: add resume
    all_epoch_loss = 0
    start = time.time()
    network.train()

    if use_cuda:
        network.cuda()

    for epoch in range(opt.epochs):
        epoch_start = time.time()
        if epoch in [2, 4]:
            lr *= config['gamma']  # TODO: use lr_scheduel
            for param_group in optimiz.param_groups:
                param_group['lr'] *= config['gamma']
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
            total_loss = sum(losses)
            total_loss.backward()
            optimiz.step()

            all_epoch_loss += total_loss.data[0]
            current_iter = epoch * len(dataset) + step + 1
            average_loss = all_epoch_loss / current_iter

            if (step+1) % config['disp_interval'] == 0:
                end = time.time()
                print('Epoch {:2d}, iter {:5d}, average loss: {:.6f}, lr: '
                      '{:.2e}'.format(epoch+1, step+1, average_loss, lr))
                print('>>>> rpn_cls: {:.6f}'.format(losses[0].data[0]))
                print('>>>> rpn_box: {:.6f}'.format(losses[1].data[0]))
                print('>>>> cls: {:.6f}'.format(losses[2].data[0]))
                print('>>>> box: {:.6f}'.format(losses[3].data[0]))
                print('>>>> reid: {:.6f}'.format(losses[4].data[0]))
                print('time cost: {:.3f}s/iter'.format(
                    (end - start) / (epoch * len(dataset) + (step + 1))))

        epoch_end = time.time()
        print('\nEntire epoch time cost: {:.2f} hours\n'.format(
            (epoch_end - epoch_start) / 3600))

        save_name = os.path.join(save_dir, 'sipn_{}.pth'.format(epoch))
        torch.save(network.state_dict(), save_name)



if __name__ == '__main__':

    main()









