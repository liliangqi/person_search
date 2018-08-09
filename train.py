# -----------------------------------------------------
# Train Spatial Invariant Person Search Network
#
# Author: Liangqi Li
# Creating Date: Mar 31, 2018
# Latest rectified: Aug 7, 2018
# -----------------------------------------------------
import os
import argparse

import torch
from torch.utils.data import DataLoader
import yaml
import time

from __init__ import clock_non_return
from dataset.sipn_dataset import SIPNDataset
import dataset.sipn_transforms as sipn_transforms
from model import SIPN


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--net', default='res50', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--out_dir', default='./output', type=str)
    parser.add_argument('--pre_model', default='', type=str)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--dataset_name', default='prw', type=str)

    args = parser.parse_args()

    return args


def train_model(dataloader, net, lr, optimizer, num_epochs, save_dir,
                resume_epoch):
    """Train the model"""

    all_epoch_loss = 0
    dataset_len = len(dataloader.dataset)
    start = time.time()
    net.train()

    with open('config.yml', 'r') as f:
        config = yaml.load(f)

    for epoch in range(resume_epoch, num_epochs):
        epoch_start = time.time()
        if epoch in [4, 8]:
            lr *= config['gamma']  # TODO: use lr_scheduel
            for param_group in optimizer.param_groups:
                param_group['lr'] *= config['gamma']

        for step, data in enumerate(dataloader):
            im, (gt_boxes, im_info) = data
            im = im.to(device)
            gt_boxes = gt_boxes.squeeze(0).to(device)
            im_info = im_info.numpy().squeeze(0)

            losses = net(im, gt_boxes, im_info)
            optimizer.zero_grad()
            total_loss = sum(losses)
            total_loss.backward()
            optimizer.step()

            all_epoch_loss += total_loss.item()
            current_iter = epoch * dataset_len + step + 1
            average_loss = all_epoch_loss / current_iter

            torch.cuda.empty_cache()

            if (step+1) % config['disp_interval'] == 0:
                end = time.time()
                print('Epoch {:2d}, iter {:5d}, average loss: {:.6f}, lr: '
                      '{:.2e}'.format(epoch+1, step+1, average_loss, lr))
                print('>>>> rpn_cls: {:.6f}'.format(losses[0].item()))
                print('>>>> rpn_box: {:.6f}'.format(losses[1].item()))
                print('>>>> cls: {:.6f}'.format(losses[2].item()))
                print('>>>> box: {:.6f}'.format(losses[3].item()))
                print('>>>> reid: {:.6f}'.format(losses[4].item()))
                print('time cost: {:.3f}s/iter'.format(
                    (end - start) / (epoch * dataset_len + (step + 1))))

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
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = os.path.join(opt.out_dir, opt.dataset_name)
    print('Trained models will be save to', os.path.abspath(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pre_model = opt.pre_model
    if opt.resume != 0:
        pre_model = ''

    model = SIPN(opt.net, opt.dataset_name, pre_model)
    model.to(device)

    if opt.resume:
        resume = os.path.join(save_dir, 'sipn_' + opt.net + '_' +
                              str(opt.resume) + '.pth')
        print('Resuming model check point from {}'.format(resume))
        model.load_trained_model(torch.load(resume))

    # Read the configuration file
    with open('config.yml', 'r') as f:
        config = yaml.load(f)
    target_size = config['target_size']
    max_size = config['max_size']
    pixel_means = config['pixel_means']

    # Compose transformations for the dataset
    transform = sipn_transforms.Compose([
        sipn_transforms.RandomHorizontalFlip(),
        sipn_transforms.Scale(target_size, max_size),
        sipn_transforms.ToTensor(),
        sipn_transforms.Normalize(pixel_means)
    ])

    # Load the dataset
    dataset = SIPNDataset(opt.data_dir, opt.dataset_name, 'train', transform)
    dataloader = DataLoader(dataset, shuffle=True, num_workers=8)

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
    train_model(dataloader, model, lr, optimizer, opt.epochs, save_dir,
                opt.resume)

    print('Done.\n')


if __name__ == '__main__':

    main()
