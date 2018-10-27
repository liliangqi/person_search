# -----------------------------------------------------
# Person Search Architecture -- DenseNet
#
# Author: Liangqi Li
# Creating Date: Apr 28, 2018
# Latest rectifying: Oct 27, 2018
# -----------------------------------------------------
import torch.nn as nn
from torchvision import models
import yaml


class DenseNet:

    def __init__(self, num_layers=121, pre_model=None):

        if num_layers == 121:
            self.net_conv_channels = 512
            self.fc7_channels = 1024
            if not pre_model:
                self.model = models.densenet121().features
            elif pre_model == 'official':
                self.model = models.densenet121(True).features
            else:
                raise NotImplementedError('No such pre-trained model.')
        elif num_layers == 161:
            self.net_conv_channels = 1056
            self.fc7_channels = 2208
            if not pre_model:
                self.model = models.densenet161().features
            elif pre_model == 'official':
                self.model = models.densenet161(True).features
            else:
                raise NotImplementedError('No such pre-trained model.')
        else:
            raise KeyError(num_layers)

        with open('config.yml', 'r') as f:
            config = yaml.load(f)
        self.fixed_blocks = config['dense121_fixed_blocks']

        self.head, self.tail = self.initialize(self.fixed_blocks)

    def initialize(self, fixed_blocks):
        for p in self.model.norm0.parameters():
            p.requires_grad = False
        for p in self.model.conv0.parameters():
            p.requires_grad = False

        assert 0 <= fixed_blocks < 4
        if fixed_blocks >= 3:
            for p in self.model.denseblock3.parameters():
                p.requires_grad = False
            for p in self.model.transition3.parameters():
                p.requires_grad = False
        if fixed_blocks >= 2:
            for p in self.model.denseblock2.parameters():
                p.requires_grad = False
            for p in self.model.transition2.parameters():
                p.requires_grad = False
        if fixed_blocks >= 1:
            for p in self.model.denseblock1.parameters():
                p.requires_grad = False
            for p in self.model.transition1.parameters():
                p.requires_grad = False

        def set_bn_fix(m):
            class_name = m.__class__.__name__
            if class_name.find('BatchNorm') != -1:
                for param in m.parameters():
                    param.requires_grad = False

        self.model.apply(set_bn_fix)

        head = nn.Sequential(self.model.conv0, self.model.norm0,
                             self.model.relu0, self.model.denseblock1,
                             self.model.transition1, self.model.denseblock2,
                             self.model.transition2, self.model.denseblock3,
                             self.model.transition3)
        tail = nn.Sequential(self.model.denseblock4, self.model.norm5)

        return head, tail

    def train(self, mode):
        if mode:
            # Set fixed blocks to be in eval mode (not really doing anything)
            self.model.eval()
            if self.fixed_blocks <= 3:
                self.model.denseblock4.train()
            if self.fixed_blocks <= 2:
                self.model.denseblock3.train()
                self.model.transition3.train()
            if self.fixed_blocks <= 1:
                self.model.denseblock2.train()
                self.model.transition2.train()
            if self.fixed_blocks == 0:
                self.model.denseblock1.train()
                self.model.transition1.train()

            # Set BatchNorm always in eval mode during training
            def set_bn_eval(m):
                class_name = m.__class__.__name__
                if class_name.find('BatchNorm') != -1:
                    m.eval()

            self.model.apply(set_bn_eval)
