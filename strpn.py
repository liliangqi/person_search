# -----------------------------------------------------
# Spatial Transformer RPN of Person Search Architecture
#
# Author: Liangqi Li and Xinlei Chen
# Creating Date: Apr 2, 2018
# Latest rectified: Apr 9, 2018
# -----------------------------------------------------
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import numpy.random as npr

from generate_anchors import generate_anchors
from losses import smooth_l1_loss
from bbox_transform import bbox_transform, bbox_transform_inv, clip_boxes, \
    bbox_overlaps
from nms.pth_nms import pth_nms as nms


class STRPN(nn.Module):

    def __init__(self, net_conv_channels, num_pid, training=True):
        """
        create Spatial Transformer Region Proposal Network
        ---
        param:
            net_conv_channels: (int) channels of feature maps extracted by head
            training: (bool) training mode or test mode
        """
        super().__init__()
        with open('config.yml', 'r') as f:
            self.config = yaml.load(f)
        self.num_pid = num_pid
        self.training = training
        self.feat_stride = self.config['rpn_feat_stride']
        self.rpn_channels = self.config['rpn_channels']
        self.anchor_scales = self.config['anchor_scales']
        self.anchor_ratios = self.config['anchor_ratios']
        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)
        self.anchors = None  # to be set in other methods

        self.rpn_net = nn.Conv2d(
            net_conv_channels, self.rpn_channels, 3, padding=1)
        self.rpn_cls_score_net = nn.Conv2d(
            self.rpn_channels, self.num_anchors * 2, 1)
        self.rpn_bbox_pred_net = nn.Conv2d(
            self.rpn_channels, self.num_anchors * 4, 1)

        self.initialize_weight(False)

    def forward(self, head_features, gt_boxes, im_info):
        rois, rpn_info, label, bbox_info = self.region_proposal(
            head_features, gt_boxes, im_info)
        rpn_label, rpn_bbox_info, rpn_cls_score, rpn_bbox_pred = rpn_info

        rpn_cls_score = rpn_cls_score.view(-1, 2)
        rpn_label = rpn_label.view(-1)
        rpn_select = Variable((rpn_label.data != -1)).nonzero().view(-1)
        rpn_cls_score = rpn_cls_score.index_select(
            0, rpn_select).contiguous().view(-1, 2)
        rpn_label = rpn_label.index_select(0, rpn_select).contiguous().view(-1)

        rpn_cls_loss = F.cross_entropy(rpn_cls_score, rpn_label)
        rpn_box_loss = smooth_l1_loss(rpn_bbox_pred, rpn_bbox_info, sigma=3.0,
                                      dim=[1, 2, 3])
        rpn_loss = (rpn_cls_loss, rpn_box_loss)

        # TODO: add roi-pooling
        pooled_feat = self.pooling(head_features, rois, max_pool=False)

        return pooled_feat, rpn_loss, label, bbox_info

    def pooling(self, bottom, rois, max_pool=True):
        rois = rois.detach()
        x1 = rois[:, 1::4] / 16.0
        y1 = rois[:, 2::4] / 16.0
        x2 = rois[:, 3::4] / 16.0
        y2 = rois[:, 4::4] / 16.0

        height = bottom.size(2)
        width = bottom.size(3)

        # affine theta
        theta = Variable(rois.data.new(rois.size(0), 2, 3).zero_())
        theta[:, 0, 0] = (x2 - x1) / (width - 1)
        theta[:, 0, 2] = (x1 + x2 - width + 1) / (width - 1)
        theta[:, 1, 1] = (y2 - y1) / (height - 1)
        theta[:, 1, 2] = (y1 + y2 - height + 1) / (height - 1)

        pooling_size = self.config['pooling_size']
        if max_pool:
            pre_pool_size = pooling_size * 2
            grid = F.affine_grid(theta, torch.Size(
                (rois.size(0), 1, pre_pool_size, pre_pool_size)))
            crops = F.grid_sample(
                bottom.expand(rois.size(0), bottom.size(1), bottom.size(2),
                              bottom.size(3)), grid)
            crops = F.max_pool2d(crops, 2, 2)
        else:
            grid = F.affine_grid(theta, torch.Size(
                (rois.size(0), 1, pooling_size, pooling_size)))
            crops = F.grid_sample(
                bottom.expand(rois.size(0), bottom.size(1), bottom.size(2),
                              bottom.size(3)), grid)

        return crops

    def anchor_compose(self, height, width):
        anchors = generate_anchors(ratios=np.array(self.anchor_ratios),
                                   scales=np.array(self.anchor_scales))
        A = anchors.shape[0]
        shift_x = np.arange(0, width) * self.feat_stride
        shift_y = np.arange(0, height) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
                            shift_y.ravel())).transpose()
        K = shifts.shape[0]
        # width changes faster, so here it is H, W, C
        anchors = anchors.reshape((1, A, 4)) + shifts.reshape(
            (1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
        length = np.int32(anchors.shape[0])

        return Variable(torch.from_numpy(anchors).cuda())

    def region_proposal(self, net_conv, gt_boxes, im_info):
        self.anchors = self.anchor_compose(net_conv.size(2), net_conv.size(3))
        rpn = F.relu(self.rpn_net(net_conv))
        rpn_cls_score = self.rpn_cls_score_net(rpn)
        rpn_cls_score_reshape = rpn_cls_score.view(
            1, 2, -1, rpn_cls_score.size()[-1])
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape)  # TODO: dim
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
            rpn_labels, rpn_bbox_info = self.anchor_target_layer(
                rpn_cls_score, gt_boxes, im_info)
            rois, label, bbox_info = self.proposal_target_layer(
                rois, roi_scores, gt_boxes)

            rpn_info = (rpn_labels, rpn_bbox_info, rpn_cls_score_reshape,
                        rpn_bbox_pred)

            return rois, rpn_info, label, bbox_info
        else:
            rois, _ = self.proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info)
            return rois

    def proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, im_info):
        if self.training:
            pre_nms_top_n = self.config['train_rpn_pre_nms_top_n']
            post_nms_top_n = self.config['train_rpn_post_nms_top_n']
            nms_thresh = self.config['train_rpn_nms_thresh']
        else:
            pre_nms_top_n = self.config['test_rpn_nms_thresh']
            post_nms_top_n = self.config['test_rpn_post_nms_top_n']
            nms_thresh = self.config['test_rpn_nms_thresh']

        # Get the scores and bounding boxes
        scores = rpn_cls_prob[:, :, :, self.num_anchors:]
        rpn_bbox_pred = rpn_bbox_pred.view((-1, 4))
        scores = scores.contiguous().view(-1, 1)
        proposals = bbox_transform_inv(self.anchors, rpn_bbox_pred)
        proposals = clip_boxes(proposals, im_info[:2])

        # Pick the top region proposals
        scores, order = scores.view(-1).sort(descending=True)
        if pre_nms_top_n > 0:
            order = order[:pre_nms_top_n]
            scores = scores[:pre_nms_top_n].view(-1, 1)
        proposals = proposals[order.data, :]

        # Non-maximal suppression
        keep = nms(torch.cat((proposals, scores), 1).data, nms_thresh)

        # Pick th top region proposals after NMS
        if post_nms_top_n > 0:
            keep = keep[:post_nms_top_n]
        proposals = proposals[keep, :]
        scores = scores[keep,]

        # Only support single image as input
        batch_inds = Variable(
            proposals.data.new(proposals.size(0), 1).zero_())
        blob = torch.cat((batch_inds, proposals), 1)

        return blob, scores

    def anchor_target_layer(self, rpn_cls_score, gt_boxes, im_info):

        def _unmap(data, count, inds, fill=0):
            """
            Unmap a subset of item (data) back to the original set of items
            (of size count)
            """
            if len(data.shape) == 1:
                ret = np.empty((count,), dtype=np.float32)
                ret.fill(fill)
                ret[inds] = data
            else:
                ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
                ret.fill(fill)
                ret[inds, :] = data
            return ret

        def _compute_targets(ex_rois, gt_rois):
            """Compute bounding-box regression targets for an image."""

            assert ex_rois.shape[0] == gt_rois.shape[0]
            assert ex_rois.shape[1] == 4
            assert gt_rois.shape[1] >= 5

            # add float convert
            return bbox_transform(torch.from_numpy(ex_rois),
                                  torch.from_numpy(gt_rois[:, :4])).numpy()

        all_anchors = self.anchors.data.cpu().numpy()
        gt_boxes = gt_boxes.data.cpu().numpy()
        rpn_cls_score = rpn_cls_score.data

        A = self.num_anchors
        total_anchors = all_anchors.shape[0]
        K = total_anchors / self.num_anchors

        # allow boxes to sit over the edge by a small amount
        _allowed_border = 0

        # map of shape (..., H, W)
        height, width = rpn_cls_score.shape[1:3]

        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -_allowed_border) &
            (all_anchors[:, 1] >= -_allowed_border) &
            (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
            (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
        )[0]

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside),), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
        if not self.config['train_rpn_clobber_positive']:
            # assign bg labels first so that positive labels can clobber them
            # first set the negatives
            labels[max_overlaps < self.config['train_rpn_neg_overlap']] = 0

            # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= self.config['train_rpn_pos_overlap']] = 1

        if self.config['train_rpn_clobber_positive']:
            # assign bg labels last so that negative labels can clobber pos
            labels[max_overlaps < self.config['train_rpn_neg_overlap']] = 0

        # subsample positive labels if we have too many
        num_fg = int(self.config['train_rpn_fg_frac'] *
                     self.config['train_rpn_batchsize'])
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = self.config['train_rpn_batchsize'] - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        # only the positive ones have regression targets
        bbox_inside_weights[labels == 1, :] = np.array(
            self.config['train_rpn_bbox_inside_weights'])

        bbox_outside_weights = np.zeros((len(inds_inside), 4),
                                        dtype=np.float32)
        if self.config['train_rpn_pos_weight'] < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((self.config['train_rpn_pos_weight'] > 0) &
                    (self.config['train_rpn_pos_weight'] < 1))
            positive_weights = (self.config['train_rpn_pos_weight'] /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - self.config['train_rpn_pos_weight']) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors,
                                     inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors,
                                      inds_inside, fill=0)

        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))
        rpn_labels = Variable(torch.from_numpy(labels).float().cuda()).long()

        # bbox_targets
        bbox_targets = bbox_targets.reshape((1, height, width, A * 4))

        rpn_bbox_targets = Variable(
            torch.from_numpy(bbox_targets).float().cuda())
        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights.reshape(
            (1, height, width, A * 4))
        rpn_bbox_inside_weights = Variable(
            torch.from_numpy(bbox_inside_weights).float().cuda())

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights.reshape(
            (1, height, width, A * 4))
        rpn_bbox_outside_weights = Variable(
            torch.from_numpy(bbox_outside_weights).float().cuda())

        return rpn_labels, (rpn_bbox_targets, rpn_bbox_inside_weights,
                            rpn_bbox_outside_weights)

    def proposal_target_layer(self, rpn_rois, rpn_scores, gt_boxes):
        """
        Assign object detection proposals to ground-truth targets. Produces
        proposal classification labels and bounding-box regression targets.
        """

        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source

        def _get_bbox_regression_labels(bbox_target_data, num_classes):
            """Bounding-box regression targets (bbox_target_data) are stored in
            a compact form N x (class, tx, ty, tw, th)

            This function expands those targets into the 4-of-4*K
            representation used by the network (i.e. only one class
            has non-zero targets).

            Returns:
                bbox_target (ndarray): N x 4K blob of regression targets
                bbox_inside_weights (ndarray): N x 4K blob of loss weights
            """
            # Inputs are tensor

            clss = bbox_target_data[:, 0]
            bbox_tar = clss.new(clss.numel(), 4 * num_classes).zero_()
            bbox_in_weights = clss.new(bbox_tar.shape).zero_()
            inds = (clss > 0).nonzero().view(-1)
            if inds.numel() > 0:
                clss = clss[inds].contiguous().view(-1, 1)
                dim1_inds = inds.unsqueeze(1).expand(inds.size(0), 4)
                dim2_inds = torch.cat(
                    [4 * clss, 4 * clss + 1, 4 * clss + 2, 4 * clss + 3],
                    1).long()
                bbox_tar[dim1_inds, dim2_inds] = bbox_target_data[inds][:, 1:]
                tr_bb_in_wei = self.config['train_bbox_inside_weights']
                bbox_in_weights[dim1_inds, dim2_inds] = bbox_tar.new(
                    tr_bb_in_wei).view(-1, 4).expand_as(dim1_inds)

            return bbox_tar, bbox_in_weights

        def _compute_targets(ex_rois, gt_rois, label):
            """Compute bounding-box regression targets for an image."""
            # Inputs are tensor

            assert ex_rois.shape[0] == gt_rois.shape[0]
            assert ex_rois.shape[1] == 4
            assert gt_rois.shape[1] == 4

            targets = bbox_transform(ex_rois, gt_rois)
            if self.config['train_bbox_normalize_targets_precomputed']:
                # Optionally normalize targets by a precomputed mean and stdev
                means = self.config['train_bbox_normalize_means']
                stds = self.config['train_bbox_normalize_stds']
                targets = ((targets - targets.new(means)) / targets.new(stds))
            return torch.cat([label.unsqueeze(1), targets], 1)

        def _sample_rois(al_rois, al_scores, gt_box, fg_rois_per_im,
                         rois_per_im, num_classes, num_pid):
            """Generate a random sample of RoIs comprising foreground and
            background examples.
            """
            # overlaps: (rois x gt_boxes)
            overlaps = bbox_overlaps(
                al_rois[:, 1:5].data,
                gt_box[:, :4].data)
            max_overlaps, gt_assignment = overlaps.max(1)
            label = gt_box[gt_assignment, [4]]

            # Select foreground RoIs as those with >= FG_THRESH overlap
            fg_inds = (max_overlaps >=
                       self.config['train_fg_thresh']).nonzero().view(-1)
            # Guard against when an image has fewer than fg_rois_per_image

            # # ========================added=======================
            # # foreground RoIs
            # fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size(0))
            # # Sample foreground regions without replacement
            # if fg_inds.size(0) > 0:
            #   fg_inds = fg_inds[torch.from_numpy(
            #     npr.choice(np.arange(0, fg_inds.numel()), size=int(
            # fg_rois_per_this_image), replace=False)).long().cuda()]
            # # ====================================================

            # Select bg RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = ((max_overlaps < self.config['train_bg_thresh_hi']) +
                       (max_overlaps >= self.config['train_bg_thresh_lo'])
                       == 2).nonzero().view(-1)

            # =========================origin==========================
            # Small modification to the original version where we ensure a
            # fixed number of regions are sampled
            if fg_inds.numel() > 0 and bg_inds.numel() > 0:
                fg_rois_per_im = min(fg_rois_per_im, fg_inds.numel())
                fg_inds = fg_inds[torch.from_numpy(
                    npr.choice(np.arange(0, fg_inds.numel()),
                               size=int(fg_rois_per_im),
                               replace=False)).long().cuda()]
                fg_inds = torch.from_numpy(
                    (np.sort(fg_inds.cpu().numpy()))).long().cuda()
                bg_rois_per_im = rois_per_im - fg_rois_per_im
                to_replace = bg_inds.numel() < bg_rois_per_im
                bg_inds = bg_inds[torch.from_numpy(
                    npr.choice(np.arange(0, bg_inds.numel()),
                               size=int(bg_rois_per_im),
                               replace=to_replace)).long().cuda()]
            elif fg_inds.numel() > 0:
                to_replace = fg_inds.numel() < rois_per_im
                fg_inds = fg_inds[torch.from_numpy(
                    npr.choice(np.arange(0, fg_inds.numel()),
                               size=int(rois_per_im),
                               replace=to_replace)).long().cuda()]
                fg_rois_per_im = rois_per_im
            elif bg_inds.numel() > 0:
                to_replace = bg_inds.numel() < rois_per_im
                bg_inds = bg_inds[torch.from_numpy(
                    npr.choice(np.arange(0, bg_inds.numel()),
                               size=int(rois_per_im),
                               replace=to_replace)).long().cuda()]
                fg_rois_per_im = 0
            else:
                import pdb
                pdb.set_trace()

            # # ====================rectify========================
            # # Compute number of background RoIs to take from this image
            # # (guarding against there being fewer than desired)
            # bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
            # bg_rois_per_this_image = min(bg_rois_per_this_image,
            # bg_inds.size(0))
            # # Sample background regions without replacement
            # if bg_inds.size(0) > 0:
            #   bg_inds = bg_inds[torch.from_numpy(
            #     npr.choice(np.arange(0, bg_inds.numel()),
            # size=int(bg_rois_per_this_image), replace=False)).long().cuda()]

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)
            # Select sampled values from various arrays:
            label = label[keep_inds].contiguous()
            # Clamp labels for the background RoIs to 0
            label[int(fg_rois_per_im):] = 0
            roi = al_rois[keep_inds].contiguous()
            roi_score = al_scores[keep_inds].contiguous()

            p_label = None
            if gt_box.size(1) > 5:
                p_label = gt_box[gt_assignment, [5]]
                p_label = p_label[keep_inds].contiguous()
                p_label[fg_rois_per_im:] = num_pid

            bbox_target_data = _compute_targets(
                roi[:, 1:5].data,
                gt_box[gt_assignment[keep_inds]][:, :4].data, label.data)

            bbox_tar, bbox_in_weights = _get_bbox_regression_labels(
                bbox_target_data, num_classes)

            return label, roi, roi_score, bbox_tar, bbox_in_weights, p_label

        _num_classes = 2
        all_rois = rpn_rois
        all_scores = rpn_scores

        # Include ground-truth boxes in the set of candidate rois
        # if cfg.TRAIN.USE_GT:
        zeros = rpn_rois.data.new(gt_boxes.size(0), 1)
        all_rois = torch.cat(
            (Variable(torch.cat((zeros, gt_boxes.data[:, :4]), 1)),
             all_rois), 0)
        # this may be a mistake, but all_scores is redundant anyway
        all_scores = torch.cat((all_scores, Variable(zeros)), 0)

        num_images = 1
        rois_per_image = self.config['train_batch_size'] / num_images
        fg_rois_per_image = int(round(
            self.config['train_fg_frac'] * rois_per_image))

        # Sample rois with classification labels and bounding box regression
        # targets
        labels, rois, roi_scores, bbox_targets, bbox_inside_weights, \
            pid_label = _sample_rois(all_rois, all_scores, gt_boxes,
                                     fg_rois_per_image, rois_per_image,
                                     _num_classes, self.num_pid)

        rois = rois.view(-1, 5)
        assert rois.size(0) == 128
        roi_scores = roi_scores.view(-1)  # TODO: remove this
        labels = labels.view(-1, 1)
        bbox_targets = bbox_targets.view(-1, _num_classes * 4)
        bbox_inside_weights = bbox_inside_weights.view(-1, _num_classes * 4)
        bbox_outside_weights = (bbox_inside_weights > 0).float()
        pid_label = pid_label.view(-1, 1)
        labels = labels.long()
        pid_label = pid_label.long()

        return rois, (labels, pid_label), (Variable(bbox_targets),
                                           Variable(bbox_inside_weights),
                                           Variable(bbox_outside_weights))

    def initialize_weight(self, trun):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            if truncated:
                # not a perfect approximation
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
            else:
                m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

        normal_init(self.rpn_net, 0, 0.01, trun)
        normal_init(self.rpn_cls_score_net, 0, 0.01, trun)
        normal_init(self.rpn_bbox_pred_net, 0, 0.01, trun)
