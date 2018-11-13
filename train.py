# -----------------------------------------------------
# Train Spatial Invariant Person Search Network
#
# Author: Liangqi Li
# Creating Date: Mar 31, 2018
# Latest rectified: Nov 5, 2018
# -----------------------------------------------------
import os
import time
import shutil

import yaml
import argparse
import torch
import torch.nn.functional as func
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from utils.utils import clock_non_return, AverageMeter
from utils.logger import TensorBoardLogger
from dataset.sipn_dataset import SIPNDataset, sipn_fn, \
    PersonSearchTripletSampler, PersonSearchTripletFn
import dataset.sipn_transforms as sipn_transforms
from models.model import SIPN
from utils.losses import TripletLoss, oim_loss


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--net', default='res50', type=str,
                        help='Network Backbone')
    parser.add_argument('--max_epoch', default=20, type=int,
                        help='Max epoch to train the model')
    parser.add_argument('--data_dir', default='', type=str,
                        help='The root path to the dataset')
    parser.add_argument('--dataset_name', default='prw', type=str,
                        help='The dataset name, `sysu` or `prw`')
    parser.add_argument('--tensorboard_dir', default='./logs/TensorBoard',
                        help='The path to save TensorBoard files', type=str)
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='Initializing learning rate.')
    parser.add_argument('--step_size', default=[7, 14], nargs='+', type=int,
                        help='Epoch steps to decay the learning rate')
    parser.add_argument('--optimizer', default='SGD', type=str,
                        help='The optimizer using for the model')
    parser.add_argument('--out_dir', default='./output', type=str,
                        help='The path to the saved models')
    parser.add_argument('--pre_model', default='', type=str,
                        help='The path to the pre-trained model, or set as '
                             '`official` to use the official one')
    parser.add_argument('--resume', default=0, type=int,
                        help='Epoch step to resume training the model')
    parser.add_argument('--loss', default='oim', type=str,
                        help='The loss to train the model, `oim` or `tri`')

    args = parser.parse_args()

    return args


def train_model(dataloader, net, optimizer, epoch, criterion):
    """Train the model"""

    lr = optimizer.param_groups[0]['lr']
    data_time_end = time.time()
    total_time_end = time.time()
    with open('config.yml', 'r') as f:
        config = yaml.load(f)

    for iter_idx, data in enumerate(dataloader):
        im, gt_boxes, im_info = data

        if isinstance(im, tuple):
            assert isinstance(gt_boxes, tuple)
            assert isinstance(im_info, tuple)
            im = tuple([x.to(device) for x in im])
            gt_boxes = tuple([x.to(device) for x in gt_boxes])
            q_im, p_im, n_im = im
            q_box, p_boxes, n_boxes = gt_boxes
            q_info, p_info, n_info = im_info
            pid = int(q_box[:, -1].item())

            data_time.update(time.time() - data_time_end)
            train_time_end = time.time()

            q_feat = net(q_im, q_box, q_info, mode='query')
            p_det_loss, p_feat, p_label = net(p_im, p_boxes, p_info)
            n_det_loss, n_feat, n_label = net(n_im, n_boxes, n_info)

            del q_box, p_boxes, n_boxes, gt_boxes

            q_feat = func.normalize(q_feat)
            p_feat = func.normalize(p_feat)
            n_feat = func.normalize(n_feat)

            p_mask = (p_label.squeeze() != net.num_pid).nonzero(
                ).squeeze().view(-1)
            p_label = p_label[p_mask]
            p_feat = p_feat[p_mask]
            n_mask = (n_label.squeeze() != net.num_pid).nonzero(
                ).squeeze().view(-1)
            n_label = n_label[n_mask]
            n_feat = n_feat[n_mask]

            tri_label = torch.cat((p_label, n_label)).squeeze()
            tri_feat = torch.cat((p_feat, n_feat), 0)
            reid_loss = criterion(q_feat, pid, tri_feat, tri_label)

            del q_feat, p_feat, n_feat
            del p_label, n_label

            losses = [x + y for x, y in zip(p_det_loss, n_det_loss)]
            losses.append(reid_loss)
        else:
            im = im.to(device)
            gt_boxes = gt_boxes.squeeze(0).to(device)
            im_info = im_info.ravel()

            data_time.update(time.time() - data_time_end)
            train_time_end = time.time()

            det_loss, feat, label = net(im, gt_boxes, im_info)
            feat = func.normalize(feat)
            reid_loss = oim_loss(feat, label, net.lut, net.queue,
                                 gt_boxes.size(0), net.lut_momentum)
            losses = list(det_loss)
            losses.append(reid_loss)

        # Backward
        optimizer.zero_grad()
        sum_loss = sum(losses)
        sum_loss.backward()
        optimizer.step()

        # Compute average loss and average time over all iterations
        current_loss = sum_loss.item()
        total_loss.update(current_loss)
        train_time.update(time.time() - train_time_end)

        # Show status
        if (iter_idx + 1) % config['disp_interval'] == 0:
            print('Epoch {:2d}, iter {:5d}, average loss: {:.6f}, lr: '
                  '{:.2e}'.format(epoch+1, iter_idx+1, total_loss.avg, lr))
            print('>>>> rpn_cls: {:.6f}'.format(losses[0].item()))
            print('>>>> rpn_box: {:.6f}'.format(losses[1].item()))
            print('>>>> cls: {:.6f}'.format(losses[2].item()))
            print('>>>> box: {:.6f}'.format(losses[3].item()))
            print('>>>> reid: {:.6f}'.format(losses[4].item()))
            print('Data Average time: {:.3f}s/iter'.format(data_time.avg))
            print('Training Average time: {:.3f}s/iter'.format(train_time.avg))
            print('Total Average time: {:.3f}s/iter'.format(total_time.avg))

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
        total_time.update(time.time() - total_time_end)
        data_time_end = time.time()
        total_time_end = time.time()


@clock_non_return
def main():

    opt = parse_args()
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(1024)

    save_dir = os.path.join(opt.out_dir, opt.dataset_name)
    print('Trained models will be saved to {}\n'.format(
        os.path.abspath(save_dir)))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Use TensorBoard to save visual results
    global tensor_logger
    tensorboard_dir = opt.tensorboard_dir
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if os.listdir(tensorboard_dir):  # Remove early TensorBoard files
        shutil.rmtree(tensorboard_dir)
        os.makedirs(tensorboard_dir)
    tensor_logger = TensorBoardLogger(tensorboard_dir)

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
    if opt.loss == 'tri':
        sampler = PersonSearchTripletSampler(dataset)
        collate_fn = PersonSearchTripletFn(dataset, sampler.batch_pids)
        dataloader = DataLoader(
            dataset, batch_sampler=sampler, collate_fn=collate_fn)
    elif opt.loss == 'oim':
        collate_fn = sipn_fn
        dataloader = DataLoader(
            dataset, shuffle=True, collate_fn=collate_fn, num_workers=8)
    else:
        raise KeyError(opt.loss)

    # Choose parameters to be updated during training
    lr = opt.lr
    params = []
    # print('These parameters will be updated during training:')
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            # print(key)
            # TODO: set different decay for weight and bias
            params += [{'params': [value], 'lr': lr, 'weight_decay': 1e-4}]

    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=0.9)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params)
    else:
        raise KeyError(opt.optimizer)

    global total_loss
    global data_time
    global train_time
    global total_time
    start_epoch = opt.resume
    criterion = TripletLoss()
    total_loss = AverageMeter()
    data_time = AverageMeter()
    train_time = AverageMeter()
    total_time = AverageMeter()

    if opt.resume:
        resume = os.path.join(save_dir, 'sipn_{}_{}.tar'.format(
            opt.net, opt.resume))
        print('Resuming model checkpoint from {}\n'.format(resume))
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

        train_model(dataloader, model, optimizer, epoch, criterion)
        scheduler.step()
        try:
            collate_fn.called_times = 0
        except AttributeError:
            pass

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
