import numpy as np
import torch
from PIL import Image 
from pymic.transform.intensity import NonLinearTransform
from pymic.transform.simclr_crop import SimCLR_RandomResizedCrop
from pymic.transform.crop import RandomResizedCrop
import copy
import random
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

np.random.seed(0)

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

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


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        crop_trans=SimCLR_RandomResizedCrop()
        image1=crop_trans(copy.deepcopy(x))
        image2=crop_trans(copy.deepcopy(x))

        slice1=torch.tensor(image1['image'])
        slice2=torch.tensor(image2['image'])
        slice_aug1=self.base_transform(slice1)
        slice_aug2=self.base_transform(slice2)
        
        x['image'] = [slice_aug1,slice_aug2]
        return x
    
    
