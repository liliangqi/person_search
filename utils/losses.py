# -----------------------------------------------------
# Custom Losses used in Person Search Architecture
#
# Author: Liangqi Li and Tong Xiao
# Creating Date: Jan 3, 2018
# Latest rectifying: Oct 25, 2018
# -----------------------------------------------------
import torch
from torch.autograd import Function, Variable
import torch.nn.functional as func
import numpy as np


def smooth_l1_loss(bbox_pred, bbox_info, sigma=1., dim=(1,)):
    sigma_2 = sigma ** 2
    bbox_targets, bbox_inside_weights, bbox_outside_weights = bbox_info
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smooth_l1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (
                sigma_2 / 2.) * smooth_l1_sign + (
                              abs_in_box_diff - (0.5 / sigma_2)) * (
                              1. - smooth_l1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box


class OIM(Function):
    def __init__(self, lut, queue, num_gt, momentum):
        super(OIM, self).__init__()
        self.lut = lut
        self.queue = queue
        self.momentum = momentum  # TODO: use exponentially weighted average
        self.num_gt = num_gt

    def forward(self, *inputs):
        inputs, targets = inputs
        self.save_for_backward(inputs, targets)
        outputs_labeled = inputs.mm(self.lut.t())
        outputs_unlabeled = inputs.mm(self.queue.t())
        return torch.cat((outputs_labeled, outputs_unlabeled), 1)

    def backward(self, *grad_outputs):
        grad_outputs, = grad_outputs
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat((self.lut, self.queue), 0))

        for i, (x, y) in enumerate(zip(inputs, targets)):
            if y == -1:
                tmp = torch.cat((self.queue[1:], x.view(1, -1)), 0)
                self.queue[:, :] = tmp[:, :]
            elif y < len(self.lut):
                if i < self.num_gt:
                    self.lut[y] = self.momentum * self.lut[y] + \
                                  (1. - self.momentum) * x
                    self.lut[y] /= self.lut[y].norm()
            else:
                continue

        return grad_inputs, None


def oim_loss(reid_feat, aux_label, lut, queue, num_gt, momentum=0.5):
    num_pid = lut.size(0)
    aux_label_np = aux_label.data.cpu().numpy()
    invalid_inds = np.where((aux_label_np < 0) | (aux_label_np >= num_pid))
    aux_label_np[invalid_inds] = -1
    pid_label = Variable(torch.from_numpy(aux_label_np).long().cuda()).view(-1)
    aux_label = aux_label.view(-1)

    reid_result = OIM(lut, queue, num_gt, momentum)(reid_feat, aux_label)
    reid_loss_weight = torch.cat([torch.ones(num_pid).cuda(),
                                  torch.zeros(queue.size(0)).cuda()])
    scalar = 10
    reid_loss = func.cross_entropy(reid_result * scalar, pid_label,
                                   weight=reid_loss_weight, ignore_index=-1)
    return reid_loss
