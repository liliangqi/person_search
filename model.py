# -----------------------------------------------------
# Person Search Model
#
# Author: Liangqi Li and Xinlei Chen
# Creating Date: Apr 1, 2018
# Latest rectifying: Apr 2, 2018
# -----------------------------------------------------
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from resnet import resnet
from strpn import STRPN
from losses import oim_loss, smooth_l1_loss


class SIPN(nn.Module):

    def __init__(self, net_name, state_dict=None, training=True):
        super().__init__()
        self.net_name = net_name
        self.training = training
        # TODO: set depending on dataset
        self.num_pid = 5532
        self.queue_size = 5000
        self.lut_momentum = 0.5
        self.reid_feat_dim = 256

        self.register_buffer('lut', torch.zeros(
            self.num_pid, self.reid_feat_dim).cuda())
        self.register_buffer('queue', torch.zeros(
            self.queue_size, self.reid_feat_dim).cuda())

        if self.net_name == 'res50':
            self.net = resnet(50, state_dict, self.training)
        else:
            raise KeyError(self.net_name)

        self.fc7_channels = self.net.fc7_channels

        # SPIN consists of three main parts
        self.head = self.net.head
        self.strpn = STRPN(self.net.net_conv_channels)
        self.tail = self.net.tail

        self.cls_score_net = nn.Linear(self.fc7_channels, 2)
        self.bbox_pred_net = nn.Linear(self.fc7_channels, 8)
        self.reid_feat_net = nn.Linear(self.fc7_channels, self.reid_feat_dim)
        self.init_linear_weight(True)

    def forward(self, im_data, gt_boxes, im_info):
        net_conv = self.head(im_data)
        # returned parameters contain 3 tuples here
        pooled_feat, rpn_loss, label, bbox_info = self.strpn(net_conv, im_info)
        fc7 = self.tail(pooled_feat)
        cls_score = self.cls_score_net(fc7)
        bbox_pred = self.bbox_pred_net(fc7)
        reid_feat = F.normalize(self.reid_feat_net(fc7))

        cls_pred = torch.max(cls_score, 1)[1]
        cls_prob = F.softmax(cls_score)
        det_label, pid_label = label

        cls_loss = F.cross_entropy(cls_score.view(-1, 2), det_label)
        bbox_loss = smooth_l1_loss(bbox_pred, bbox_info)
        reid_loss = oim_loss(reid_feat, pid_label, self.num_pid,
                             self.queue_size, self.lut,
                             self.queue, self.lut_momentum)
        rpn_cls_loss, rpn_box_loss = rpn_loss

        return rpn_cls_loss, rpn_box_loss, cls_loss, bbox_loss, reid_loss

    def train(self, mode=True):
        nn.Module.train(self, mode)
        self.net.train(mode)

    def init_linear_weight(self, trun):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
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