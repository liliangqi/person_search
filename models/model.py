# -----------------------------------------------------
# Spatial Invariant Person Search Network
#
# Author: Liangqi Li and Xinlei Chen
# Creating Date: Apr 1, 2018
# Latest rectified: Nov 5, 2018
# -----------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as func
import yaml

from .vgg16 import Vgg16
from .resnet import MyResNet
from .densenet import DenseNet
from .strpn import STRPN
from utils.losses import oim_loss, smooth_l1_loss, TripletLoss


class SIPN(nn.Module):

    def __init__(self, net_name, dataset_name, pre_model=''):
        super().__init__()
        self.net_name = net_name

        if dataset_name == 'sysu':
            self.num_pid = 5532
            self.queue_size = 5000
        elif dataset_name == 'prw':
            self.num_pid = 483
            self.queue_size = 500
        else:
            raise KeyError(dataset_name)
        self.lut_momentum = 0.5
        self.reid_feat_dim = 256

        self.register_buffer('lut', torch.zeros(
            self.num_pid, self.reid_feat_dim).cuda())
        self.register_buffer('queue', torch.zeros(
            self.queue_size, self.reid_feat_dim).cuda())

        if self.net_name == 'vgg16':
            self.net = Vgg16(pre_model)
        elif self.net_name == 'res34':
            self.net = MyResNet(34, pre_model)
        elif self.net_name == 'res50':
            self.net = MyResNet(50, pre_model)
        elif self.net_name == 'dense121':
            self.net = DenseNet(121, pre_model)
        elif self.net_name == 'dense161':
            self.net = DenseNet(161, pre_model)
        else:
            raise KeyError(self.net_name)

        self.fc7_channels = self.net.fc7_channels

        # SPIN consists of three main parts
        self.head = self.net.head
        self.strpn = STRPN(self.net.net_conv_channels, self.num_pid)
        self.tail = self.net.tail

        self.cls_score_net = nn.Linear(self.fc7_channels, 2)
        self.bbox_pred_net = nn.Linear(self.fc7_channels, 8)
        self.reid_feat_net = nn.Linear(self.fc7_channels, self.reid_feat_dim)
        self.init_linear_weight(False)

        self.triplet_loss = TripletLoss()

    def forward(self, im_data, gt_boxes, im_info, mode='gallery'):
        if self.training:
            # ###############################################################
            # ========================Triplet Loss===========================
            # ###############################################################
            if isinstance(im_data, tuple):
                assert isinstance(gt_boxes, tuple)
                assert isinstance(im_info, tuple)

                # Extract the feature of the query
                q_im = im_data[0]
                q_box = gt_boxes[0]
                q_info = im_info[0]

                q_box = torch.cat((torch.zeros(1, 1).cuda(), q_box[:, :4]), 1)
                q_net_conv = self.head(q_im)
                q_pool_feat = self.strpn(q_net_conv, q_box, q_info, 'query')
                if self.net_name == 'vgg16':
                    q_pool_feat = q_pool_feat.view(q_pool_feat.size(0), -1)
                    q_fc7 = self.tail(q_pool_feat)
                else:
                    q_fc7 = self.tail(q_pool_feat).mean(3).mean(2)
                q_reid_feat = self.reid_feat_net(q_fc7)

                # Extract the feature of the positive gallery
                p_im = im_data[1]
                p_boxes = gt_boxes[1]
                p_info = im_info[1]

                p_net_conv = self.head(p_im)
                p_pool_feat, p_tr_feat, p_rpn_loss, p_label, p_bbox_info = \
                    self.strpn(p_net_conv, p_boxes, p_info)
                if self.net_name == 'vgg16':
                    p_pool_feat = p_pool_feat.view(p_pool_feat.size(0), -1)
                    p_fc7 = self.tail(p_pool_feat)
                else:
                    p_fc7 = self.tail(p_pool_feat).mean(3).mean(2)
                p_cls_score = self.cls_score_net(p_fc7)
                p_bbox_pred = self.bbox_pred_net(p_fc7)
                p_reid_feat = self.reid_feat_net(p_fc7)

                p_det_label, p_pid_label = p_label
                p_det_label = p_det_label.view(-1)
                p_cls_loss = func.cross_entropy(
                    p_cls_score.view(-1, 2), p_det_label)
                p_bbox_loss = smooth_l1_loss(p_bbox_pred, p_bbox_info)
                p_rpn_cls_loss, p_rpn_box_loss = p_rpn_loss

                # Extract the feature of the negative gallery
                n_im = im_data[2]
                n_boxes = gt_boxes[2]
                n_info = im_info[2]

                n_net_conv = self.head(n_im)
                n_pool_feat, n_tr_feat, n_rpn_loss, n_label, n_bbox_info = \
                    self.strpn(n_net_conv, n_boxes, n_info)
                if self.net_name == 'vgg16':
                    n_pool_feat = n_pool_feat.view(n_pool_feat.size(0), -1)
                    n_fc7 = self.tail(n_pool_feat)
                else:
                    n_fc7 = self.tail(n_pool_feat).mean(3).mean(2)
                n_cls_score = self.cls_score_net(n_fc7)
                n_bbox_pred = self.bbox_pred_net(n_fc7)
                n_reid_feat = self.reid_feat_net(n_fc7)

                n_det_label, n_pid_label = n_label
                n_det_label = n_det_label.view(-1)
                n_cls_loss = func.cross_entropy(
                    n_cls_score.view(-1, 2), n_det_label)
                n_bbox_loss = smooth_l1_loss(n_bbox_pred, n_bbox_info)
                n_rpn_cls_loss, n_rpn_box_loss = n_rpn_loss

                # Compute loss
                rpn_cls_loss = p_rpn_cls_loss + n_rpn_cls_loss
                rpn_box_loss = p_rpn_box_loss + n_rpn_box_loss
                cls_loss = p_cls_loss + n_cls_loss
                bbox_loss = p_bbox_loss + n_bbox_loss

                query_pid = int(gt_boxes[0][:, -1].item())
                p_mask = (p_pid_label.squeeze() != self.num_pid).nonzero(
                    ).squeeze().view(-1)
                p_pid_label_drop = p_pid_label[p_mask]
                p_reid_feat_drop = p_reid_feat[p_mask]
                n_mask = (n_pid_label.squeeze() != self.num_pid).nonzero(
                    ).squeeze().view(-1)
                n_pid_label_drop = n_pid_label[n_mask]
                n_reid_feat_drop = n_reid_feat[n_mask]

                tri_label = torch.cat(
                    (p_pid_label_drop, n_pid_label_drop)).squeeze()
                tri_feat = torch.cat((p_reid_feat_drop, n_reid_feat_drop), 0)
                reid_loss = self.triplet_loss(
                    q_reid_feat, query_pid, tri_feat, tri_label, mode='hard')

                return rpn_cls_loss, rpn_box_loss, cls_loss, bbox_loss,\
                    reid_loss

            # ###############################################################
            # ###############################################################

            net_conv = self.head(im_data)
            # Returned parameters contain 3 tuples here
            pooled_feat, trans_feat, rpn_loss, label, bbox_info = self.strpn(
                net_conv, gt_boxes, im_info)
            if self.net_name == 'vgg16':
                pooled_feat = pooled_feat.view(pooled_feat.size(0), -1)
                fc7 = self.tail(pooled_feat)
            else:
                fc7 = self.tail(pooled_feat).mean(3).mean(2)
            cls_score = self.cls_score_net(fc7)
            bbox_pred = self.bbox_pred_net(fc7)

            # reid_fc7 = self.tail(trans_feat).mean(3).mean(2)
            # reid_feat = F.normalize(self.reid_feat_net(reid_fc7))
            reid_feat = func.normalize(self.reid_feat_net(fc7))

            det_label, pid_label = label
            det_label = det_label.view(-1)
            cls_loss = func.cross_entropy(cls_score.view(-1, 2), det_label)
            bbox_loss = smooth_l1_loss(bbox_pred, bbox_info)
            reid_loss = oim_loss(reid_feat, pid_label, self.lut, self.queue,
                                 gt_boxes.size(0), self.lut_momentum)
            rpn_cls_loss, rpn_box_loss = rpn_loss

            return rpn_cls_loss, rpn_box_loss, cls_loss, bbox_loss, reid_loss

        else:
            if mode == 'gallery':
                net_conv = self.head(im_data)
                rois, pooled_feat, trans_feat = self.strpn(
                    net_conv, gt_boxes, im_info)
                if self.net_name == 'vgg16':
                    pooled_feat = pooled_feat.view(pooled_feat.size(0), -1)
                    fc7 = self.tail(pooled_feat)
                else:
                    fc7 = self.tail(pooled_feat).mean(3).mean(2)
                cls_score = self.cls_score_net(fc7)
                bbox_pred = self.bbox_pred_net(fc7)

                # reid_fc7 = self.tail(trans_feat).mean(3).mean(2)
                # reid_feat = F.normalize(self.reid_feat_net(reid_fc7))
                reid_feat = func.normalize(self.reid_feat_net(fc7))

                cls_prob = func.softmax(cls_score, 1)

                with open('config.yml', 'r') as f:
                    config = yaml.load(f)
                mean = config['train_bbox_normalize_means']
                std = config['train_bbox_normalize_stds']
                means = bbox_pred.new(mean).repeat(2).unsqueeze(0).expand_as(
                    bbox_pred)
                stds = bbox_pred.new(std).repeat(2).unsqueeze(0).expand_as(
                    bbox_pred)
                bbox_pred = bbox_pred.mul(stds).add(means)

                cls_prob = cls_prob.cpu().numpy()
                bbox_pred = bbox_pred.cpu().numpy()
                rois = rois.cpu().numpy()
                reid_feat = reid_feat.cpu().numpy()

                return cls_prob, bbox_pred, rois, reid_feat

            elif mode == 'query':
                net_conv = self.head(im_data)
                # TODO: move pooling layer from strpn to SIPN
                pooled_feat = self.strpn(net_conv, gt_boxes, im_info, mode)
                if self.net_name == 'vgg16':
                    pooled_feat = pooled_feat.view(pooled_feat.size(0), -1)
                    fc7 = self.tail(pooled_feat)
                else:
                    fc7 = self.tail(pooled_feat).mean(3).mean(2)
                reid_feat = func.normalize(self.reid_feat_net(fc7))

                return reid_feat.data.cpu().numpy()

            else:
                raise KeyError(mode)

    def train(self, mode=True):
        nn.Module.train(self, mode)
        self.net.train(mode)

    def init_linear_weight(self, trun):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initializer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

        normal_init(self.cls_score_net, 0, 0.01, trun)
        normal_init(self.bbox_pred_net, 0, 0.001, trun)
        # TODO: change 0.01 for reid_feat_net
        normal_init(self.reid_feat_net, 0, 0.01, trun)

    def load_trained_model(self, state_dict):
        nn.Module.load_state_dict(
            self, {k: state_dict[k] for k in list(self.state_dict())})
