# -----------------------------------------------------
# Person Search Architecture -- Vgg16
#
# Author: Liangqi Li
# Creating Date: May 6, 2018
# Latest rectifying: Oct 27, 2018
# -----------------------------------------------------
import torch.nn as nn
import torchvision.models as models


class Vgg16:

    def __init__(self, pre_model=None):
        self.net_conv_channels = 512
        self.fc7_channels = 4096

        if not pre_model:
            self.model = models.vgg16()
        elif pre_model == 'official':
            self.model = models.vgg16(True)
        else:
            raise NotImplementedError('No such pre-trained model.')

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
