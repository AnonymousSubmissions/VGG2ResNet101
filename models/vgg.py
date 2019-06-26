# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init
import time
baselayer = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
        512, 512, 512]



def weights_init(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.kaiming_normal(m.state_dict()[key], mode='fan_out')
            if 'bn' in key:
                m.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [
        pool5, conv6,
        nn.ReLU(inplace=True), conv7,
        nn.ReLU(inplace=True)
    ]
    return layers


class VGG16Extractor(nn.Module):
    def __init__(self):
        super(VGG16Extractor, self).__init__()
        self.vgg = nn.ModuleList(vgg(baselayer, 3))
        self._init_modules()

    def _init_modules(self):
        self.vgg.apply(weights_init)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        # conv block 1
        torch.cuda.synchronize()
        t1 = time.time()
        for k in range(4):
            x = self.vgg[k](x)
        torch.cuda.synchronize()
        t2 = time.time()

        # conv block 2
        for k in range(4, 9):
            x = self.vgg[k](x)
        torch.cuda.synchronize()
        t3 = time.time()

        # conv block 3
        for k in range(9, 16):
            x = self.vgg[k](x)
        torch.cuda.synchronize()
        t4 = time.time()

        # conv block 4
        for k in range(16, 23):
            x = self.vgg[k](x)
        torch.cuda.synchronize()
        t5 = time.time()

        # conv block 5
        for k in range(23, 30):
            x = self.vgg[k](x)
        torch.cuda.synchronize()
        t6 = time.time()

        # dilated conv
        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        torch.cuda.synchronize()
        t7 = time.time()
        return x, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6
