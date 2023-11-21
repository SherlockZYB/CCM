# -*- coding: utf-8 -*-
from __future__ import print_function, division
import copy 
import json
import math
import random
import numpy as np
import torch
from torch.nn.functional import pad,interpolate
from pymic.transform.abstract_transform import AbstractTransform
from pymic.util.image_process import *
from scipy.ndimage import zoom
from torchvision.utils import make_grid
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def tensor_shuffle(input):

    return input[torch.randperm(len(input),device=input.device)]

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

class IntensityClip(AbstractTransform):
    """
    Clip the intensity for input image

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `IntensityClip_channels`: (list) A list of int for specifying the channels.
    :param `IntensityClip_lower`: (list) The lower bound for clip in each channel.
    :param `IntensityClip_upper`: (list) The upper bound for clip in each channel.
    :param `IntensityClip_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `False`.
    """
    def __init__(self, params):
        super(IntensityClip, self).__init__(params)
        self.channels =  params['IntensityClip_channels'.lower()]
        self.lower = params.get('IntensityClip_lower'.lower(), None)
        self.upper = params.get('IntensityClip_upper'.lower(), None)
        self.inverse   = params.get('IntensityClip_inverse'.lower(), False)
    
    def __call__(self, sample):
        image = sample['image']
        lower = self.lower if self.lower is not None else [None] * len(self.channels)
        upper = self.upper if self.upper is not None else [None] * len(self.channels)
        for chn in self.channels:
            lower_c, upper_c = lower[chn], upper[chn]
            if(lower_c is None):
                lower_c = np.percentile(image[chn], 0.05)
            if(upper_c is None):
                upper_c = np.percentile(image[chn, 99.95])
            image[chn] = np.clip(image[chn], lower_c, upper_c)
        sample['image'] = image
        return sample
    
class GammaCorrection(AbstractTransform):
    """
    Apply random gamma correction to given channels.

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `GammaCorrection_channels`: (list) A list of int for specifying the channels.
    :param `GammaCorrection_gamma_min`: (float) The minimal gamma value.
    :param `GammaCorrection_gamma_max`: (float) The maximal gamma value.
    :param `GammaCorrection_probability`: (optional, float) 
        The probability of applying GammaCorrection. Default is 0.5.
    :param `GammaCorrection_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `False`.
    """
    def __init__(self, params):
        super(GammaCorrection, self).__init__(params)
        self.channels =  params['GammaCorrection_channels'.lower()]
        self.gamma_min = params['GammaCorrection_gamma_min'.lower()]
        self.gamma_max = params['GammaCorrection_gamma_max'.lower()]
        self.prob      = params.get('GammaCorrection_probability'.lower(), 0.5)
        self.inverse   = params.get('GammaCorrection_inverse'.lower(), False)
    
    def __call__(self, sample):
        if(np.random.uniform() > self.prob):
            return sample
        image= sample['image']
        for chn in self.channels:
            gamma_c = random.random() * (self.gamma_max - self.gamma_min) + self.gamma_min
            img_c = image[chn]
            v_min = img_c.min()
            v_max = img_c.max()
            if(v_min < v_max):
                img_c = (img_c - v_min)/(v_max - v_min)
                img_c = np.power(img_c, gamma_c)*(v_max - v_min) + v_min
            image[chn] = img_c

        sample['image'] = image
        return sample

class GaussianNoise(AbstractTransform):
    """
    Add Gaussian Noise to given channels.

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `GaussianNoise_channels`: (list) A list of int for specifying the channels.
    :param `GaussianNoise_mean`: (float) The mean value of noise.
    :param `GaussianNoise_std`: (float) The std of noise.
    :param `GaussianNoise_probability`: (optional, float) 
        The probability of applying GaussianNoise. Default is 0.5.
    :param `GaussianNoise_inverse`: (optional, bool) 
        Is inverse transform needed for inference. Default is `False`.
    """
    def __init__(self, params):
        super(GaussianNoise, self).__init__(params)
        self.channels = params['GaussianNoise_channels'.lower()]
        self.mean     = params['GaussianNoise_mean'.lower()]
        self.std      = params['GaussianNoise_std'.lower()]
        self.prob     = params.get('GaussianNoise_probability'.lower(), 0.5)
        self.inverse  = params.get('GaussianNoise_inverse'.lower(), False)
    
    def __call__(self, sample):
        if(np.random.uniform() > self.prob):
            return sample
        image= sample['image']
        for chn in self.channels:
            img_c = image[chn]
            noise = np.random.normal(self.mean, self.std, img_c.shape)
            image[chn] = img_c + noise

        sample['image'] = image
        return sample

class GrayscaleToRGB(AbstractTransform):
    """
    Convert gray scale images to RGB by copying channels. 
    """
    def __init__(self, params):
        super(GrayscaleToRGB, self).__init__(params)
        self.inverse = params.get('GrayscaleToRGB_inverse'.lower(), False)
    
    def __call__(self, sample):
        image= sample['image']
        assert(image.shape[0] == 1 or image.shape[0] == 3)
        if(image.shape[0] == 1):
            sample['image'] = np.concatenate([image, image, image])
        return sample
    
class NonLinearTransform(AbstractTransform):
    def __init__(self, params):
        super(NonLinearTransform, self).__init__(params)
        self.inverse  = params.get('NonLinearTransform_inverse'.lower(), False)
        self.prob     = params.get('NonLinearTransform_probability'.lower(), 0.5)
    
    def __call__(self, sample):
        if(random.random() >  self.prob):
            return sample

        image= sample['image'] 
        points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
        xvals, yvals = bezier_curve(points, nTimes=100000)
        if random.random() < 0.5: # Half change to get flip
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        image = np.interp(image, xvals, yvals)
        sample['image']  = image 
        return sample

class LocalShuffling(AbstractTransform):
    """
    local pixel shuffling of an input image, used for self-supervised learning 
    """
    def __init__(self, params):
        super(LocalShuffling, self).__init__(params)
        self.inverse  = params.get('LocalShuffling_inverse'.lower(), False)
        self.prob     = params.get('LocalShuffling_probability'.lower(), 0.5)
        self.block_range = params.get('LocalShuffling_block_range'.lower(), (5000, 10000))
        self.block_size_min = params.get('LocalShuffling_block_size_min'.lower(), None)
        self.block_size_max = params.get('LocalShuffling_block_size_max'.lower(), None)

    def __call__(self, sample):
        if(random.random() >  self.prob):
            return sample

        image= sample['image']       
        img_shape = image.shape
        img_dim = len(img_shape) - 1
        assert(img_dim == 2 or img_dim == 3)
        img_out = copy.deepcopy(image)
        if(self.block_size_min is None):
            block_size_min = [2] * img_dim
        elif(isinstance(self.block_size_min, int)):
            block_size_min = [self.block_size_min] * img_dim
        else:
            assert(len(self.block_size_min) == img_dim)
            block_size_min = self.block_size_min

        if(self.block_size_max is None):
            block_size_max = [img_shape[1+i]//10 for i in range(img_dim)]
        elif(isinstance(self.block_size_min, int)):
            block_size_max = [self.block_size_max] * img_dim
        else:
            assert(len(self.block_size_max) == img_dim)
            block_size_max = self.block_size_max
        block_num = random.randint(self.block_range[0], self.block_range[1])

        for n in range(block_num):
            block_size = [random.randint(block_size_min[i], block_size_max[i]) \
                for i in range(img_dim)]    
            coord_min = [random.randint(0, img_shape[1+i] - block_size[i]) \
                for i in range(img_dim)]
            if(img_dim == 2):
                window = image[:, coord_min[0]:coord_min[0] + block_size[0], 
                                  coord_min[1]:coord_min[1] + block_size[1]]
                n_pixels = block_size[0] * block_size[1]
            else:
                window = image[:, coord_min[0]:coord_min[0] + block_size[0], 
                                  coord_min[1]:coord_min[1] + block_size[1],
                                  coord_min[2]:coord_min[2] + block_size[2]]
                n_pixels = block_size[0] * block_size[1] * block_size[2]
            window = np.reshape(window, [-1, n_pixels])
            np.random.shuffle(np.transpose(window))
            window = np.transpose(window)
            if(img_dim == 2):
                window = np.reshape(window, [-1, block_size[0], block_size[1]])
                img_out[:, coord_min[0]:coord_min[0] + block_size[0], 
                           coord_min[1]:coord_min[1] + block_size[1]] = window
            else:
                window = np.reshape(window, [-1, block_size[0], block_size[1], block_size[2]])
                img_out[:, coord_min[0]:coord_min[0] + block_size[0], 
                           coord_min[1]:coord_min[1] + block_size[1],
                           coord_min[2]:coord_min[2] + block_size[2]] = window
        sample['image'] = img_out
        return sample

class InPainting(AbstractTransform):
    """
    In-painting of an input image, used for self-supervised learning 
    """
    def __init__(self, params):
        super(InPainting, self).__init__(params)
        self.inverse  = params.get('InPainting_inverse'.lower(), False)
        self.prob     = params.get('InPainting_probability'.lower(), 0.5)
        self.block_range = params.get('InPainting_block_range'.lower(), (1, 6))
        self.block_size_min = params.get('InPainting_block_size_min'.lower(), None)
        self.block_size_max = params.get('InPainting_block_size_max'.lower(), None)

    def __call__(self, sample):
        if(random.random() >  self.prob):
            return sample

        image= sample['image']       
        img_shape = image.shape
        img_dim = len(img_shape) - 1
        assert(img_dim == 2 or img_dim == 3)

        if(self.block_size_min is None):
            block_size_min = [img_shape[1+i]//6 for i in range(img_dim)]
        elif(isinstance(self.block_size_min, int)):
            block_size_min = [self.block_size_min] * img_dim
        else:
            assert(len(self.block_size_min) == img_dim)
            block_size_min = self.block_size_min

        if(self.block_size_max is None):
            block_size_max = [img_shape[1+i]//3 for i in range(img_dim)]
        elif(isinstance(self.block_size_min, int)):
            block_size_max = [self.block_size_max] * img_dim
        else:
            assert(len(self.block_size_max) == img_dim)
            block_size_max = self.block_size_max
        block_num = random.randint(self.block_range[0], self.block_range[1])

        for n in range(block_num):
            block_size = [random.randint(block_size_min[i], block_size_max[i]) \
                for i in range(img_dim)]    
            coord_min = [random.randint(0, img_shape[1+i] - block_size[i]) \
                for i in range(img_dim)]
            if(img_dim == 2):
                random_block = np.random.rand(img_shape[0], block_size[0], block_size[1])
                image[:, coord_min[0]:coord_min[0] + block_size[0], 
                         coord_min[1]:coord_min[1] + block_size[1]] = random_block
            else:
                random_block = np.random.rand(img_shape[0], block_size[0], 
                                              block_size[1], block_size[2])
                image[:, coord_min[0]:coord_min[0] + block_size[0], 
                         coord_min[1]:coord_min[1] + block_size[1],
                         coord_min[2]:coord_min[2] + block_size[2]] = random_block
        sample['image'] = image
        return sample

class OutPainting(AbstractTransform):
    """
    Out-painting of an input image, used for self-supervised learning 
    """
    def __init__(self, params):
        super(OutPainting, self).__init__(params)
        self.inverse  = params.get('OutPainting_inverse'.lower(), False)
        self.prob     = params.get('OutPainting_probability'.lower(), 0.5)
        self.block_range = params.get('OutPainting_block_range'.lower(), (1, 6))
        self.block_size_min = params.get('OutPainting_block_size_min'.lower(), None)
        self.block_size_max = params.get('OutPainting_block_size_max'.lower(), None)

    def __call__(self, sample):
        if(random.random() >  self.prob):
            return sample

        image= sample['image']       
        img_shape = image.shape
        img_dim = len(img_shape) - 1
        assert(img_dim == 2 or img_dim == 3)
        img_out = np.random.rand(*img_shape)

        if(self.block_size_min is None):
            block_size_min = [img_shape[1+i] - 4 * img_shape[1+i]//7 for i in range(img_dim)]
        elif(isinstance(self.block_size_min, int)):
            block_size_min = [self.block_size_min] * img_dim
        else:
            assert(len(self.block_size_min) == img_dim)
            block_size_min = self.block_size_min

        if(self.block_size_max is None):
            block_size_max = [img_shape[1+i] - 3 * img_shape[1+i]//7 for i in range(img_dim)]
        elif(isinstance(self.block_size_min, int)):
            block_size_max = [self.block_size_max] * img_dim
        else:
            assert(len(self.block_size_max) == img_dim)
            block_size_max = self.block_size_max
        block_num = random.randint(self.block_range[0], self.block_range[1])

        for n in range(block_num):
            block_size = [random.randint(block_size_min[i], block_size_max[i]) \
                for i in range(img_dim)]    
            coord_min = [random.randint(0, img_shape[1+i] - block_size[i]) \
                for i in range(img_dim)]
            if(img_dim == 2):
                img_out[:, coord_min[0]:coord_min[0] + block_size[0], 
                           coord_min[1]:coord_min[1] + block_size[1]] = \
                    image[:, coord_min[0]:coord_min[0] + block_size[0], 
                             coord_min[1]:coord_min[1] + block_size[1]]
            else:
                img_out[:, coord_min[0]:coord_min[0] + block_size[0], 
                           coord_min[1]:coord_min[1] + block_size[1],
                           coord_min[2]:coord_min[2] + block_size[2]] = \
                    image[:, coord_min[0]:coord_min[0] + block_size[0], 
                             coord_min[1]:coord_min[1] + block_size[1],
                             coord_min[2]:coord_min[2] + block_size[2]] 
        sample['image'] = img_out
        return sample

class InOutPainting(AbstractTransform):
    """
    Apply in-painting or out-patining randomly. They are mutually exclusive.
    """
    def __init__(self, params):
        super(InOutPainting, self).__init__(params)
        self.inverse  = params.get('InOutPainting_inverse'.lower(), False)
        self.prob     = params.get('InOutPainting_probability'.lower(), 0.5)
        self.in_prob  = params.get('InPainting_probability'.lower(), 0.5)
        params['InPainting_probability']  = 1.0
        params['outPainting_probability'] = 1.0
        self.inpaint  = InPainting(params)
        self.outpaint = OutPainting(params)

    def __call__(self, sample):
        if(random.random() >  self.prob):
            return sample
        if(random.random() < self.in_prob):
            sample = self.inpaint(sample)
        else:
            sample = self.outpaint(sample)
        return sample
    
class Masked(AbstractTransform):
    """
    Masked image modeling referring to the MAE, used for self-supervised learning 
    """
    def __init__(self, params):
        super(Masked, self).__init__(params)
        self.ratio = params.get('Masked_ratio'.lower())
        self.size  = params.get('Masked_size'.lower())
        self.prob  = params.get('Masked_prob'.lower())

    def __call__(self, sample):
        if(random.random() >  self.prob):
            return sample

        image=sample['image']
        b, c, h, w = image.shape
        p_h, p_w = h // self.size, w // self.size
        mask = np.random.rand(b, p_h, p_w)
        mask = zoom(mask.reshape(b, 1, p_h, p_w),
                    (1, 1, self.size, self.size),
                    order=0) > (1 - self.ratio)
        pad_h = h- mask.shape[2]
        pad_w = w - mask.shape[3]
        mask = np.pad(mask, ((0, 0), (0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
                    mode="constant")
        image[mask.repeat(c, 1)] = 0.0
        sample['image'] = image
        return sample
    
class Pixel_Shuffling_Masked(AbstractTransform):
    """
    Masked image modeling referring to the MAE, used for self-supervised learning 
    """
    def __init__(self, params):
        super(Pixel_Shuffling_Masked, self).__init__(params)
        self.ratio = params.get('Masked_ratio'.lower())
        self.size  = params.get('Masked_size'.lower())
        self.prob  = params.get('Masked_prob'.lower())

    def __call__(self, sample):
        if(random.random() >  self.prob):
            return sample
        
        img=torch.from_numpy(sample['image'][0])
        img = img.permute(1, 2, 0)
        h, w = img.shape[:2]

        assert h % self.size == 0 and w % self.size == 0, f"img : {h,w} self.size: {self.size}"
        # 切割img成hxw/self.size/self.size个patch size x patch size的小块
        patched_data = torch.stack([p for f in torch.chunk(img, h//self.size)
                                    for p in torch.chunk(f, w//self.size, dim=1)])
        patch_num, c = patched_data.shape[0], patched_data.shape[-1]
        num_patch_pixel = self.size*self.size

        selected_idx = torch.randperm(patch_num)[:int(patch_num*self.ratio)]
        
        selected_patch = patched_data[selected_idx]
        shuffled_patch = [tensor_shuffle(p.view(num_patch_pixel, c)).view(self.size, self.size, c) for p in selected_patch]
        patched_data[selected_idx] = torch.stack(shuffled_patch)
        shuffled_img = make_grid(patched_data.permute(
            0, 3, 1, 2), nrow=w//self.size, padding=0)
        ret = shuffled_img[0].numpy()
        ret = np.expand_dims(ret,0)
        ret = np.expand_dims(ret,0)
        sample['image']=ret
        return sample
   
    
class ConMAE_Masked(AbstractTransform):
    """
    Masked image modeling referring to the MAE, used for self-supervised learning 
    """
    def __init__(self, params):
        super(ConMAE_Masked, self).__init__(params)
        self.ratio = params.get('Masked_ratio'.lower())
        self.size  = params.get('Masked_size'.lower())
        self.prob  = params.get('Masked_prob'.lower())

    def __call__(self, sample):
        if(random.random() >  self.prob):
            return sample

        image1=copy.deepcopy(sample['image']) 
        image2=copy.deepcopy(sample['image']) 
        b, c, h, w = image1.shape
        p_h, p_w = h // self.size, w // self.size
        mask = np.random.rand(b, p_h, p_w)
        mask = zoom(mask.reshape(b, 1, p_h, p_w),
                    (1, 1, self.size, self.size),
                    order=0) > (1 - self.ratio)
        pad_h = h- mask.shape[2]
        pad_w = w - mask.shape[3]
        mask = np.pad(mask, ((0, 0), (0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
                    mode="constant")
        counter_mask = ((1-mask)>0)
        image1[mask.repeat(c, 1)] = 0.0
        image2[counter_mask.repeat(c,1)] = 0.0
        sample['image'] = np.concatenate((image1,image2),axis=0)
        return sample
    
class ConMAE_Masked_Shuffle(AbstractTransform):
    """
    Masked image modeling referring to the MAE, used for self-supervised learning 
    """
    def __init__(self, params):
        super(ConMAE_Masked_Shuffle, self).__init__(params)
        self.ratio = params.get('Masked_ratio'.lower())
        self.size  = params.get('Masked_size'.lower())
        self.prob  = params.get('Masked_prob'.lower())

    def __call__(self, sample):
        if(random.random() >  self.prob):
            return sample

        image=sample['image']
        
        img=torch.from_numpy(image[0])
        img = img.permute(1, 2, 0)
        h, w = img.shape[:2]

        assert h % self.size == 0 and w % self.size == 0, f"img : {h,w} self.size: {self.size}"
        # 切割img成hxw/self.size/self.size个patch size x patch size的小块
        patched_data = torch.stack([p for f in torch.chunk(img, h//self.size)
                                    for p in torch.chunk(f, w//self.size, dim=1)])
        counter_patched_data = torch.stack([p for f in torch.chunk(img, h//self.size)
                                    for p in torch.chunk(f, w//self.size, dim=1)])
        patch_num, c = patched_data.shape[0], patched_data.shape[-1]
        num_patch_pixel = self.size*self.size

        patch_idx = torch.randperm(patch_num)
        selected_idx = patch_idx[:int(patch_num*self.ratio)]
        counter_selected_idx = patch_idx[int(patch_num*self.ratio):]
        
        selected_patch = patched_data[selected_idx]
        counter_selected_patch = patched_data[counter_selected_idx]

        shuffled_patch = [tensor_shuffle(p.view(num_patch_pixel, c)).view(self.size, self.size, c) for p in selected_patch]
        counter_shuffled_patch = [tensor_shuffle(p.view(num_patch_pixel, c)).view(self.size, self.size, c) for p in counter_selected_patch]
        patched_data[selected_idx] = torch.stack(shuffled_patch)
        counter_patched_data[counter_selected_idx] = torch.stack(counter_shuffled_patch)

        shuffled_img = make_grid(patched_data.permute(
            0, 3, 1, 2), nrow=w//self.size, padding=0)
        counter_shuffled_img = make_grid(counter_patched_data.permute(
            0, 3, 1, 2), nrow=w//self.size, padding=0)
        
        ret_1 = shuffled_img[0].numpy()
        ret_1 = np.expand_dims(ret_1,0)
        ret_1 = np.expand_dims(ret_1,0)
        
        ret_2 = counter_shuffled_img[0].numpy()
        ret_2 = np.expand_dims(ret_2,0)
        ret_2 = np.expand_dims(ret_2,0)
        sample['image'] = np.concatenate((ret_1,ret_2),axis=0)
        return sample