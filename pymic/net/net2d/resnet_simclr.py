# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np 
from torch.nn.functional import interpolate
import torchvision.models as models


class ResNet_SimCLR(nn.Module):
    def __init__(self, params):
        super(ResNet_SimCLR, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.out_dim   = self.params['out_dim']
        self.dropout   = self.params['dropout']



        self.backbone = models.resnet50(pretrained=False, num_classes=self.out_dim)
        dim_mlp = self.backbone.fc.in_features

        self.backbone.conv1 = nn.Conv2d(self.in_chns, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def forward(self, x):
        
        return self.backbone(x)

