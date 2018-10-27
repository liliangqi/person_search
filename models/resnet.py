# -----------------------------------------------------
# Person Search Architecture -- Resnet
#
# Author: Liangqi Li and Xinlei Chen
# Creating Date: Apr 1, 2018
# Latest rectifying: Oct 27, 2018
# -----------------------------------------------------
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

import yaml


__all__ = ['ResNet', 'MyResNet', 'resnet18', 'resnet34', 'resnet50',
           'resnet101', 'resnet152']

root_url = 'https://s3.amazonaws.com/pytorch/models/'
model_urls = {
    'resnet18': root_url + 'resnet18-5c106cde.pth',
    'resnet34': root_url + 'resnet34-333f7ec4.pth',
    'resnet50': root_url + 'resnet50-19c8e357.pth',
    'resnet101': root_url + 'resnet101-5d3b4d8f.pth',
    'resnet152': root_url + 'resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride,
                               bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               # change
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # maxpool different from pytorch-resnet, to match tf-faster-rcnn
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # use stride 1 for the last conv4 layer (same as tf-faster-rcnn)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


def resnet18(pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False):
    """Constructs a ResNet-34 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False):
    """Constructs a ResNet-101 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False):
    """Constructs a ResNet-152 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class MyResNet:

    def __init__(self, num_layers=50, pre_model=None):

        if num_layers == 34:
            self.net_conv_channels = 256
            self.fc7_channels = 512
            if not pre_model:
                self.model = resnet34()
            elif pre_model == 'official':
                self.model = resnet34(True)
            else:
                raise NotImplementedError('No such pre-trained model.')
        elif num_layers == 50:
            self.net_conv_channels = 1024
            self.fc7_channels = 2048
            if not pre_model:
                self.model = resnet50()
            elif pre_model == 'official':
                self.model = resnet50(True)
            else:
                self.model = resnet50()
                state_dict = torch.load(pre_model)
                self.model.load_state_dict(
                    {k: state_dict[k] for k in list(
                        self.model.state_dict())
                     if 'num_batches_tracked' not in k})
        else:
            raise KeyError(num_layers)

        with open('config.yml', 'r') as f:
            config = yaml.load(f)
        self.fixed_blocks = config['res50_fixed_blocks']

        self.head, self.tail = self.initialize(self.fixed_blocks)

    def initialize(self, fixed_blocks):
        for p in self.model.bn1.parameters():
            p.requires_grad = False
        for p in self.model.conv1.parameters():
            p.requires_grad = False

        assert 0 <= fixed_blocks < 4
        if fixed_blocks >= 3:
            for p in self.model.layer3.parameters():
                p.requires_grad = False
        if fixed_blocks >= 2:
            for p in self.model.layer2.parameters():
                p.requires_grad = False
        if fixed_blocks >= 1:
            for p in self.model.layer1.parameters():
                p.requires_grad = False

        def set_bn_fix(m):
            class_name = m.__class__.__name__
            if class_name.find('BatchNorm') != -1:
                for param in m.parameters():
                    param.requires_grad = False

        self.model.apply(set_bn_fix)

        layer3_head = [self.model.layer3[i] for i in range(3)]
        layer3_head = nn.Sequential(*layer3_head)
        layer3_tail = [self.model.layer3[i] for i in range(3, 6)]
        layer3_tail = nn.Sequential(*layer3_tail)

        head = nn.Sequential(self.model.conv1, self.model.bn1,
                             self.model.relu, self.model.maxpool,
                             self.model.layer1, self.model.layer2,
                             layer3_head)
        tail = nn.Sequential(layer3_tail, self.model.layer4)

        return head, tail

    def train(self, mode):
        if mode:
            # Set fixed blocks to be in eval mode (not really doing anything)
            self.model.eval()
            if self.fixed_blocks <= 3:
                self.model.layer4.train()
            if self.fixed_blocks <= 2:
                self.model.layer3.train()
            if self.fixed_blocks <= 1:
                self.model.layer2.train()
            if self.fixed_blocks == 0:
                self.model.layer1.train()

            # Set batchnorm always in eval mode during training
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.model.apply(set_bn_eval)
