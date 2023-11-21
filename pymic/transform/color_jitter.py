import torch
import json
import math
import random
import numpy as np
from scipy import ndimage
from pymic import TaskType
from pymic.transform.abstract_transform import AbstractTransform
from pymic.util.image_process import *

class ColorJitter(AbstractTransform):
    
    def __init__(self, params):
        super(ColorJitter, self).__init__(params)
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