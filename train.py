# -----------------------------------------------------
# Train Spatial Invariant Person Search Network
#
# Author: Liangqi Li
# Creating Date: Mar 31, 2018
# Latest rectified: May 11, 2018
# -----------------------------------------------------
import os
import argparse

import torch
import yaml
from torch.autograd import Variable
import time

from __init__ import clock_non_return
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
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--dataset_name', default='sysu', type=str)

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


def train_model(dataset, net, lr, optimizer, num_epochs, use_cuda, save_dir,
                resume_epoch):
    """Train the model"""

    all_epoch_loss = 0
    start = time.time()
    net.train()

    if use_cuda:
        net.cuda()

    with open('config.yml', 'r') as f:
        config = yaml.load(f)

    for epoch in range(resume_epoch, num_epochs):
        epoch_start = time.time()
        if epoch in [2, 4]:
            lr *= config['gamma']  # TODO: use lr_scheduel
            for param_group in optimizer.param_groups:
                param_group['lr'] *= config['gamma']

        for step in range(len(dataset)):
            im, gt_boxes, im_info = dataset.next()
            im = im.transpose([0, 3, 1, 2])

            if use_cuda:
                im = Variable(torch.from_numpy(im).cuda())
                gt_boxes = Variable(torch.from_numpy(gt_boxes).float().cuda())
            else:
                im = Variable(torch.from_numpy(im))
                gt_boxes = Variable(torch.from_numpy(gt_boxes).float())

            losses = net(im, gt_boxes, im_info)
            optimizer.zero_grad()
            total_loss = sum(losses)
            total_loss.backward()
            optimizer.step()

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

        # Save the trained model after each epoch
        save_name = os.path.join(
            save_dir, 'sipn_{}_{}.pth'.format(net.net_name, epoch + 1))
        torch.save(net.state_dict(), save_name)


@clock_non_return
def main():

    opt = parse_args()
    use_cuda = cuda_mode(opt)

    save_dir = os.path.join(opt.out_dir, opt.dataset_name)
    print('Trained models will be save to', os.path.abspath(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pre_model = opt.pre_model
    if opt.resume != 0:
        pre_model = ''

    model = SIPN(opt.net, opt.dataset_name, pre_model)

    if opt.resume:
        resume = os.path.join(save_dir, 'sipn_' + opt.net + '_' +
                              str(opt.resume) + '.pth')
        print('Resuming model check point from {}'.format(resume))
        model.load_trained_model(torch.load(resume))

    # Load the dataset
    dataset = PersonSearchDataset(opt.data_dir, opt.dataset_name)

    # Choose parameters to be updated during training
    lr = opt.lr
    params = []
    print('These parameters will be updated during training:')
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            print(key)
            # TODO: set different decay for weight and bias
            params += [{'params': [value], 'lr': lr, 'weight_decay': 1e-4}]

    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=0.9)
    elif opt.optimizer == 'Adam':
        # lr *= 0.1
        optimizer = torch.optim.Adam(params)
    else:
        raise KeyError(opt.optimizer)

    # Train the model
    train_model(dataset, model, lr, optimizer, opt.epochs, use_cuda, save_dir,
                opt.resume)

    print('Done.\n')


if __name__ == '__main__':

    main()
