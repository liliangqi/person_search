# -----------------------------------------------------
# Person Search Architecture -- Vgg16
#
# Author: Liangqi Li
# Creating Date: May 6, 2018
# Latest rectifying: May 6, 2018
# -----------------------------------------------------
import torch
import torch.nn as nn
import torchvision.models as models


class Vgg16:

    def __init__(self, pre_model=None, training=True):
        self.net_conv_channels = 512
        self.fc7_channels = 4096
        self.training = training

        if self.training:
            if pre_model:
                self.model = models.vgg16()
                state_dict = torch.load(pre_model)
                self.model.load_state_dict({k: state_dict[k] for k in list(
                    self.model.state_dict())})
            else:
                self.model = models.vgg16(True)
        else:
            self.model = models.vgg16()

        self.head, self.tail = self.initialize()

    def initialize(self):
        for layer in range(10):
            for p in self.model.features[layer].parameters():
                p.requires_grad = False

        head = nn.Sequential(*list(self.model.features._modules.values())[:-1])
        tail = nn.Sequential(*list(
            self.model.classifier._modules.values())[:-1])

        return head, tail

    def train(self, mode):
        pass