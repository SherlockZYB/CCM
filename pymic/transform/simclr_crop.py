# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import json
import math
import random
import numpy as np
from scipy import ndimage
from pymic import TaskType
from pymic.transform.abstract_transform import AbstractTransform
from pymic.util.image_process import *
from pymic.transform.crop import CenterCrop

class SimCLR_RandomResizedCrop(CenterCrop):
    """
    Randomly resize and crop the input image (shape [C, D, H, W]). 
    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `RandomResizedCrop_output_size`: (list/tuple) Desired output size [D, H, W].
        The output channel is the same as the input channel. 
    :param `RandomResizedCrop_scale_range`: (list/tuple) Range of scale, e.g. (0.08, 1.0).
    :param `RandomResizedCrop_inverse`: (optional, bool) Is inverse transform needed for inference.
        Default is `False`. Currently, the inverse transform is not supported, and 
        this transform is assumed to be used only during training stage. 
    """
    def __init__(self):
        self.output_size = [192,192]
        self.scale       = [0.8,1]
        self.inverse     = False
        self.task        = 'self_sup'
        assert isinstance(self.output_size, (list, tuple))
        assert isinstance(self.scale, (list, tuple))
        
    def __call__(self, sample):
        image = sample['image']
        #从所有层中随机抽取一张
        slice_idx = random.randint(0,image.shape[1]-1)
        image = image[0][slice_idx]
        image = np.expand_dims(image,0)
        channel, input_size = image.shape[0], image.shape[1:]
        input_dim   = len(input_size)
        assert(input_dim == len(self.output_size))
        scale = self.scale[0] + random.random()*(self.scale[1] - self.scale[0])
        crop_size = [int(self.output_size[i] * scale)  for i in range(input_dim)]
        crop_margin = [input_size[i] - crop_size[i] for i in range(input_dim)]
        pad_image = False
        if(min(crop_margin) < 0):
            pad_image = True
            pad_size = [max(0, -crop_margin[i]) for  i in range(input_dim)]
            pad_lower = [int(pad_size[i] / 2) for i in range(input_dim)]
            pad_upper = [pad_size[i] - pad_lower[i] for i in range(input_dim)]
            pad = [(pad_lower[i], pad_upper[i]) for  i in range(input_dim)]
            pad = tuple([(0, 0)] + pad)
            image = np.pad(image, pad, 'reflect')
            crop_margin = [max(0, crop_margin[i]) for i in range(input_dim)]
        
        crop_min = [random.randint(0, item) for item in crop_margin]
        crop_max = [crop_min[i] + crop_size[i] for i in range(input_dim)]
        crop_min = [0] + crop_min
        crop_max = [channel] + crop_max

        image_t = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
        scale = [(self.output_size[i] + 0.0)/crop_size[i] for i in range(input_dim)]
        scale = [1.0] + scale
        image_t = ndimage.interpolation.zoom(image_t, scale, order = 1)
        sample['image'] = image_t
        
        if('label' in sample and \
            self.task in [TaskType.SEGMENTATION, TaskType.RECONSTRUCTION]):
            label = sample['label']
            if(pad_image):
                label = np.pad(label, pad, 'reflect')
            crop_max[0] = label.shape[0]
            label = crop_ND_volume_with_bounding_box(label, crop_min, crop_max)
            order = 0 if(self.task == TaskType.SEGMENTATION) else 1
            label = ndimage.interpolation.zoom(label, scale, order = order)
            sample['label'] = label
        if('pixel_weight' in sample and \
            self.task in [TaskType.SEGMENTATION, TaskType.RECONSTRUCTION]):
            weight = sample['pixel_weight']
            if(pad_image):
                weight = np.pad(weight, pad, 'reflect')
            crop_max[0] = weight.shape[0]
            weight = crop_ND_volume_with_bounding_box(weight, crop_min, crop_max)
            weight = ndimage.interpolation.zoom(weight, scale, order = 1)
            sample['pixel_weight'] = weight
        return sample