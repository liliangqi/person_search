# -----------------------------------------------------
# Person Search Model
#
# Author: Liangqi Li and Xinlei Chen
# Creating Date: Apr 1, 2018
# Latest rectifying: Apr 1, 2018
# -----------------------------------------------------
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from resnet import resnet


class SIPN(nn.Module):

    def __init__(self, net_name, state_dict=None, training=True):
        super().__init__()
        self.net_name = net_name
        self.training = training

        if self.net_name == 'res50':
            self.net = resnet(50, state_dict, self.training)
        else:
            raise KeyError(self.net_name)

        self.head = self.net.head
        self.tail = self.net.tail
        self.rpn_net = self.net.rpn_net
        self.rpn_cls_score_net = self.net.rpn_cls_score_net
        self.rpn_bbox_pred_net = self.net.rpn_bbox_pred_net

    def forward(self, im_data, gt_boxes, im_info):
        net_conv = self.head(im_data)
        rois = self.region_proposal(net_conv, im_info)

    def region_proposal(self, net_conv, im_info):
        rpn = F.relu(self.rpn_net(net_conv))
        rpn_cls_score = self.rpn_cls_score_net(rpn)
        rpn_cls_score_reshape = rpn_cls_score.view(
            1, 2, -1, rpn_cls_score.size()[-1])
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape)
        rpn_cls_prob = rpn_cls_prob_reshape.view_as(rpn_cls_score).permute(
            0, 2, 3, 1)
        rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1)
        rpn_cls_score_reshape = rpn_cls_score_reshape.permute(
            0, 2, 3, 1).contiguous()
        rpn_cls_pred = torch.max(rpn_cls_score_reshape.view(-1, 2), 1)[1]
        rpn_bbox_pred = self.rpn_bbox_pred_net(rpn)
        rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous()

        if self.training:
            rois, roi_scores = self.proposal_layer(rpn_cls_prob, rpn_bbox_pred,
                                                   im_info)
            rpn_labels = self.anchor_target_layer(rpn_cls_score)
            rois, _ = self.proposal_target_layer(rois, roi_scores)
        else:
            rois, _ = self.proposal_layer(rpn_cls_prob, rpn_bbox_pred)

        return rois

    def proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, im_info):

        return 1, 2


    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode (not really doing anything)
            self.net.model.eval()
            if self.net.fixed_blocks <= 3:
                self.net.model.layer4.train()
            if self.net.fixed_blocks <= 2:
                self.net.model.layer3.train()
            if self.net.fixed_blocks <= 1:
                self.net.model.layer2.train()
            if self.net.fixed_blocks == 0:
                self.net.model.layer1.train()

            # Set batchnorm always in eval mode during training
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.net.model.apply(set_bn_eval)
