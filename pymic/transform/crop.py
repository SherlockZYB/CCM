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

class CenterCrop(AbstractTransform):
    """
    Crop the given image at the center.
    Input shape should be [C, D, H, W] or [C, H, W].

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `CenterCrop_output_size`: (list or tuple) The output size. 
        [D, H, W] for 3D images and [H, W] for 2D images.
        If D is None, then the z-axis is not cropped.
    :param `CenterCrop_inverse`: (optional, bool) Is inverse transform needed for inference.
        Default is `True`.
    """
    def __init__(self, params):
        super(CenterCrop, self).__init__(params)
        self.output_size = params['CenterCrop_output_size'.lower()]
        self.inverse = params.get('CenterCrop_inverse'.lower(), True)

    def _get_crop_param(self, sample):
        input_shape = sample['image'].shape
        input_dim   = len(input_shape) - 1
        assert(input_dim == len(self.output_size))
        temp_output_size = self.output_size
        if(input_dim == 3 and self.output_size[0] is None):
            # note that output size is [D, H, W] and input is [C, D, H, W]
            temp_output_size = [input_shape[1]] + self.output_size[1:]

        crop_margin = [input_shape[i + 1] - temp_output_size[i]\
            for i in range(input_dim)]
        crop_min = [int(item/2) for item in crop_margin]
        crop_max = [crop_min[i] + temp_output_size[i] \
            for i in range(input_dim)]
        crop_min = [0] + crop_min
        crop_max = list(input_shape[0:1]) + crop_max
        sample['CenterCrop_Param'] = json.dumps((input_shape, crop_min, crop_max))
        return sample, crop_min, crop_max

    def __call__(self, sample):
        image = sample['image']
        sample, crop_min, crop_max = self._get_crop_param(sample)

        image_t = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
        sample['image'] = image_t
        
        if('label' in sample and \
            self.task in [TaskType.SEGMENTATION, TaskType.RECONSTRUCTION]):
            label = sample['label']
            crop_max[0] = label.shape[0]
            label = crop_ND_volume_with_bounding_box(label, crop_min, crop_max)
            sample['label'] = label
        if('pixel_weight' in sample and \
            self.task in [TaskType.SEGMENTATION, TaskType.RECONSTRUCTION]):
            weight = sample['pixel_weight']
            crop_max[0] = weight.shape[0]
            weight = crop_ND_volume_with_bounding_box(weight, crop_min, crop_max)
            sample['pixel_weight'] = weight
        return sample

    def _get_param_for_inverse_transform(self, sample):
        if(isinstance(sample['CenterCrop_Param'], list) or \
            isinstance(sample['CenterCrop_Param'], tuple)):
            params = json.loads(sample['CenterCrop_Param'][0]) 
        else:
            params = json.loads(sample['CenterCrop_Param']) 
        return params

    def inverse_transform_for_prediction(self, sample):
        params = self._get_param_for_inverse_transform(sample)
        origin_shape = params[0]
        crop_min     = params[1]
        crop_max     = params[2]
        predict = sample['predict']
        if(isinstance(predict, tuple) or isinstance(predict, list)):
            output_predict = []
            for predict_i in predict:
                origin_shape_i   = list(predict_i.shape[:2]) + origin_shape[1:]
                output_predict_i = np.zeros(origin_shape_i, predict_i.dtype)
                crop_min_i = [0, 0] + crop_min[1:]
                crop_max_i = list(predict_i.shape[:2]) + crop_max[1:]
                output_predict_i = set_ND_volume_roi_with_bounding_box_range(output_predict_i,
                    crop_min_i, crop_max_i, predict_i)
                output_predict.append(output_predict_i)
        else:
            origin_shape   = list(predict.shape[:2]) + origin_shape[1:]
            output_predict = np.zeros(origin_shape, predict.dtype)
            crop_min = [0, 0] + crop_min[1:]
            crop_max = list(predict.shape[:2]) + crop_max[1:]
            output_predict = set_ND_volume_roi_with_bounding_box_range(output_predict,
                crop_min, crop_max, predict)
        
        sample['predict'] = output_predict
        return sample

class CropWithBoundingBox(CenterCrop):
    """
    Crop the image (shape [C, D, H, W] or [C, H, W]) based on a bounding box.
    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `CropWithBoundingBox_start`: (None, or list/tuple) The start index 
        along each spatial axis. If None, calculate the start index automatically 
        so that the cropped region is centered at the non-zero region.
    :param `CropWithBoundingBox_output_size`: (None or tuple/list): 
        Desired spatial output size.
        If None, set it as the size of bounding box of non-zero region.
    :param `CropWithBoundingBox_inverse`: (optional, bool) Is inverse transform needed for inference.
        Default is `True`.
    """
    def __init__(self, params):
        self.start       = params['CropWithBoundingBox_start'.lower()]
        self.output_size = params['CropWithBoundingBox_output_size'.lower()]
        self.inverse     = params.get('CropWithBoundingBox_inverse'.lower(), True)
        self.task = params['task']
        
    def _get_crop_param(self, sample):
        image = sample['image']
        input_shape = sample['image'].shape
        input_dim   = len(input_shape) - 1
        bb_min, bb_max = get_ND_bounding_box(image)
        bb_min, bb_max = bb_min[1:], bb_max[1:]
        if(self.start is None):
            if(self.output_size is None):
                crop_min, crop_max = bb_min, bb_max
            else:
                assert(len(self.output_size) == input_dim)
                crop_min = [int((bb_min[i] + bb_max[i] + 1)/2) - int(self.output_size[i]/2) \
                    for i in range(input_dim)]
                crop_min = [max(0, crop_min[i]) for i in range(input_dim)]
                crop_max = [crop_min[i] + self.output_size[i] for i in range(input_dim)]
        else:
            assert(len(self.start) == input_dim)
            crop_min = self.start
            if(self.output_size is None):
                assert(len(self.output_size) == input_dim)
                crop_max = [crop_min[i] + bb_max[i] - bb_min[i] \
                    for i in range(input_dim)]
            else:
                crop_max =  [crop_min[i] + self.output_size[i] for i in range(input_dim)]
        crop_min = [0] + crop_min
        crop_max = list(input_shape[0:1]) + crop_max
        sample['CropWithBoundingBox_Param'] = json.dumps((input_shape, crop_min, crop_max))   
        print("for crop", crop_min, crop_max)
        return sample, crop_min, crop_max

    def _get_param_for_inverse_transform(self, sample):
        if(isinstance(sample['CropWithBoundingBox_Param'], list) or \
            isinstance(sample['CropWithBoundingBox_Param'], tuple)):
            params = json.loads(sample['CropWithBoundingBox_Param'][0]) 
        else:
            params = json.loads(sample['CropWithBoundingBox_Param']) 
        return params
        
class CropWithForeground(CenterCrop):
    """
    Crop the image (shape [C, D, H, W] or [C, H, W]) based on a bounding box.
    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `CropWithBoundingBox_start`: (None, or list/tuple) The start index 
        along each spatial axis. If None, calculate the start index automatically 
        so that the cropped region is centered at the non-zero region.
    :param `CropWithBoundingBox_output_size`: (None or tuple/list): 
        Desired spatial output size.
        If None, set it as the size of bounding box of non-zero region.
    :param `CropWithBoundingBox_inverse`: (optional, bool) Is inverse transform needed for inference.
        Default is `True`.
    """
    def __init__(self, params):
        self.labels  = params.get('CropWithForeground_labels'.lower(), None)
        self.margin  = params.get('CropWithForeground_margin'.lower(), [5, 10, 10])
        self.inverse = params.get('CropWithForeground_inverse'.lower(), True)
        self.task = params['task']
        
    def _get_crop_param(self, sample):
        image = sample['image']
        label = sample['label']
        input_shape = sample['image'].shape

        bb_min, bb_max = get_ND_bounding_box(label, margin=[0] + self.margin)
        bb_max[0] = input_shape[0]

        sample['CropWithForeground_Param'] = json.dumps((input_shape, bb_min, bb_max))   

        return sample, bb_min, bb_max

    def _get_param_for_inverse_transform(self, sample):
        if(isinstance(sample['CropWithForeground_Param'], list) or \
            isinstance(sample['CropWithForeground_Param'], tuple)):
            params = json.loads(sample['CropWithForeground_Param'][0]) 
        else:
            params = json.loads(sample['CropWithForeground_Param']) 
        return params
    
class RandomCrop(CenterCrop):
    """Randomly crop the input image (shape [C, D, H, W] or [C, H, W]).

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `RandomCrop_output_size`: (list/tuple) Desired output size [D, H, W] or [H, W].
        The output channel is the same as the input channel. 
        If D is None for 3D images, the z-axis is not cropped.
    :param `RandomCrop_foreground_focus`: (optional, bool) 
        If true, allow crop around the foreground. Default is False.
    :param `RandomCrop_foreground_ratio`: (optional, float) 
        Specifying the probability of foreground focus cropping when 
        `RandomCrop_foreground_focus` is True.
    :param `RandomCrop_mask_label`: (optional, None, or list/tuple) 
        Specifying the foreground labels for foreground focus cropping when 
        `RandomCrop_foreground_focus` is True. If it is None (by default), 
        the mask label will be the list of all the foreground classes. 
    :param `RandomCrop_inverse`: (optional, bool) Is inverse transform needed for inference.
        Default is `True`.
    """
    def __init__(self, params):
        self.output_size = params['RandomCrop_output_size'.lower()]
        self.fg_focus    = params.get('RandomCrop_foreground_focus'.lower(), False)
        self.fg_ratio    = params.get('RandomCrop_foreground_ratio'.lower(), 0.5)
        self.mask_label  = params.get('RandomCrop_mask_label'.lower(), None)
        self.inverse     = params.get('RandomCrop_inverse'.lower(), True)
        self.task        = params['Task'.lower()]
        assert isinstance(self.output_size, (list, tuple))
        if(self.mask_label is not None):
            assert isinstance(self.mask_label, (list, tuple))

    def _get_crop_param(self, sample):
        image       = sample['image']
        chns        = image.shape[0]
        input_shape = image.shape[1:]
        input_dim   = len(input_shape)
        assert(input_dim == len(self.output_size))

        crop_margin = [input_shape[i] - self.output_size[i] for i in range(input_dim)]
        crop_min = [0 if item == 0 else random.randint(0, item) for item in crop_margin]
        crop_max = [crop_min[i] + self.output_size[i] for i in range(input_dim)]
        
        if(self.fg_focus and random.random() < self.fg_ratio):
            label = sample['label'][0]
            if(self.mask_label is None):
                mask_label = np.unique(label)[1:]
            else:
                mask_label = self.mask_label
            random_label = random.choice(mask_label)
            crop_min, crop_max = get_random_box_from_mask(label == random_label, self.output_size)

        crop_min = [0] + crop_min
        crop_max = [chns] + crop_max

        sample['RandomCrop_Param'] = json.dumps((image.shape, crop_min, crop_max))
        return sample, crop_min, crop_max

    def _get_param_for_inverse_transform(self, sample):
        if(isinstance(sample['RandomCrop_Param'], list) or \
            isinstance(sample['RandomCrop_Param'], tuple)):
            params = json.loads(sample['RandomCrop_Param'][0]) 
        else:
            params = json.loads(sample['RandomCrop_Param']) 
        return params

class RandomResizedCrop(CenterCrop):
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
    def __init__(self, params):
        self.output_size = params['RandomResizedCrop_output_size'.lower()]
        self.scale       = params['RandomResizedCrop_scale_range'.lower()]
        self.inverse     = params.get('RandomResizedCrop_inverse'.lower(), False)
        self.task        = params['Task_type'.lower()]
        assert isinstance(self.output_size, (list, tuple))
        assert isinstance(self.scale, (list, tuple))
        
    def __call__(self, sample):
        image = sample['image']
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