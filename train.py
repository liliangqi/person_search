# -----------------------------------------------------
# Train Spatial Invariant Person Search Network
#
# Author: Liangqi Li
# Creating Date: Mar 31, 2018
# Latest rectified: Oct 25, 2018
# -----------------------------------------------------
import os
import time

import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from utils.utils import clock_non_return, AverageMeter
from utils.logger import TensorBoardLogger
from dataset.sipn_dataset import SIPNDataset
import dataset.sipn_transforms as sipn_transforms
from models.model import SIPN


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--net', default='res50', type=str)
    parser.add_argument('--max_epoch', default=20, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--step_size', default=[7, 14], nargs='+', type=int)
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--out_dir', default='./output', type=str)
    parser.add_argument('--pre_model', default='', type=str)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--dataset_name', default='prw', type=str)

    args = parser.parse_args()

    return args


def train_model(dataloader, net, optimizer, epoch):
    """Train the model"""

    lr = optimizer.param_groups[0]['lr']
    end = time.time()
    with open('config.yml', 'r') as f:
        config = yaml.load(f)

    for iter_idx, data in enumerate(dataloader):
        im, (gt_boxes, im_info) = data
        im = im.to(device)
        gt_boxes = gt_boxes.squeeze(0).to(device)
        im_info = im_info.numpy().squeeze(0)

        # Forward and backward
        losses = net(im, gt_boxes, im_info)
        optimizer.zero_grad()
        sum_loss = sum(losses)
        sum_loss.backward()
        optimizer.step()

        # Compute average loss and average time over all iterations
        current_loss = sum_loss.item()
        total_loss.update(current_loss)
        time_cost.update(time.time() - end)
        end = time.time()

        # Show status
        if (iter_idx + 1) % config['disp_interval'] == 0:
            print('Epoch {:2d}, iter {:5d}, average loss: {:.6f}, lr: '
                  '{:.2e}'.format(epoch+1, iter_idx+1, total_loss.avg, lr))
            print('>>>> rpn_cls: {:.6f}'.format(losses[0].item()))
            print('>>>> rpn_box: {:.6f}'.format(losses[1].item()))
            print('>>>> cls: {:.6f}'.format(losses[2].item()))
            print('>>>> box: {:.6f}'.format(losses[3].item()))
            print('>>>> reid: {:.6f}'.format(losses[4].item()))
            print('Average time: {:.3f}s/iter'.format(time_cost.avg))

        step = total_loss.count
        # TensorBoard logging
        if step % (config['disp_interval'] * 5) == 0:
            # Scalar values
            info = {'total_loss': current_loss,
                    'rpn_cls_loss': losses[0].item(),
                    'rpn_box_loss': losses[1].item(),
                    'cls_loss': losses[2].item(),
                    'box_loss': losses[3].item(),
                    'reid_loss': losses[4].item()}
            for tag, value in info.items():
                tensor_logger.scalar_summary(tag, value, step)

            # Values of parameters and gradients
            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                tensor_logger.hist_summary(tag, value.data.cpu().numpy(), step)
                if value.requires_grad:
                    if value.grad is None:
                        continue
                    tensor_logger.hist_summary(
                        tag + '/grad', value.grad.data.cpu().numpy(), step)

        torch.cuda.empty_cache()


@clock_non_return
def main():

    opt = parse_args()
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = os.path.join(opt.out_dir, opt.dataset_name)
    print('Trained models will be saved to', os.path.abspath(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pre_model = opt.pre_model
    if opt.resume != 0:
        pre_model = ''

    model = SIPN(opt.net, opt.dataset_name, pre_model)
    model.to(device)

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
        optimizer = torch.optim.Adam(params)
    else:
        raise KeyError(opt.optimizer)

    global total_loss
    global time_cost
    global tensor_logger

    start_epoch = opt.resume
    total_loss = AverageMeter()
    time_cost = AverageMeter()
    tensor_logger = TensorBoardLogger('./logs/TensorBoard')

    if opt.resume:
        resume = os.path.join(save_dir, 'sipn_' + opt.net + '_' +
                              str(opt.resume) + '.tar')
        print('Resuming model check point from {}'.format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        model.load_trained_model(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        total_loss = checkpoint['total_loss']

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.step_size,
                                         gamma=0.1, last_epoch=start_epoch-1)
    # Train the model
    for epoch in range(start_epoch, opt.max_epoch):
        epoch_start = time.time()
        model.train()

        train_model(dataloader, model, optimizer, epoch)
        scheduler.step()

        epoch_end = time.time()
        print('\nEntire epoch time cost: {:.2f} hours\n'.format(
            (epoch_end - epoch_start) / 3600))

        # Save the trained model after each epoch
        save_name = os.path.join(
            save_dir, 'sipn_{}_{}.tar'.format(model.net_name, epoch + 1))
        checkpoint = {'epoch': epoch + 1,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'total_loss': total_loss}
        torch.save(checkpoint, save_name)


if __name__ == '__main__':

    main()
