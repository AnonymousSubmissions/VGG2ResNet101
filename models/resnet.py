# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import time

def weights_init(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.kaiming_normal(m.state_dict()[key], mode='fan_out')
            if 'bn' in key:
                m.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
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


class Resnet(nn.Module):
    def __init__(self, block, num_blocks):
        super(Resnet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self._init_modules()


    def _init_modules(self):
        self.conv1.apply(weights_init)
        self.layer1.apply(weights_init)
        self.layer2.apply(weights_init)
        self.layer3.apply(weights_init)
        self.layer4.apply(weights_init)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # conv block 1
        torch.cuda.synchronize()
        t1 = time.time()
        c1 = F.relu(self.bn1(self.conv1(x)))
        torch.cuda.synchronize()
        t2 = time.time()

        # conv block 2
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        torch.cuda.synchronize()
        t3 = time.time()

        # conv block 3
        c3 = self.layer2(c2)
        torch.cuda.synchronize()
        t4 = time.time()

        # conv block 4
        c4 = self.layer3(c3)
        torch.cuda.synchronize()
        t5 = time.time()

        # conv block 5
        c5 = self.layer4(c4)
        torch.cuda.synchronize()
        t6 = time.time()
        return c5, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5


def ResnetExtractor(type):
    if type == '50':
        return Resnet(Bottleneck, [3, 4, 6, 3])
    elif type == '101':
        return Resnet(Bottleneck, [3, 4, 23, 3])
    elif type == '152':
        return Resnet(Bottleneck, [3, 8, 36, 3])
    else:
        print('no this type ...')
