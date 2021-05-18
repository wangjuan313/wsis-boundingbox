import torch
from torchvision_wj.utils.losses_func import *

# ypred: predicted network output
# ytrue: true segmentation mask
# mask:  mask with bounding box region equal to 1
# gt_boxes: bounding boxes of objects 

class CrossEntropyLoss():
    def __init__(self, mode='all', epsilon=1e-6):
        self.mode = mode
        self.epsilon = epsilon

    def __call__(self, ypred, ytrue):
        ypred = torch.clamp(ypred, self.epsilon, 1-self.epsilon)
        loss_pos = -ytrue*torch.log(ypred)
        loss_neg  = -(1-ytrue)*torch.log(1-ypred)

        loss_pos = torch.sum(loss_pos,dim=(0,2,3))
        loss_neg = torch.sum(loss_neg, dim=(0,2,3))
        nb_pos = torch.sum(ytrue,dim=(0,2,3))
        nb_neg = torch.sum(1-ytrue,dim=(0,2,3))

        if self.mode=='all':
            loss  = (loss_pos+loss_neg)/(nb_pos+nb_neg)
        elif self.mode=='balance':
            loss  = (loss_pos/nb_pos+loss_neg/nb_neg)/2
        return loss


class MILUnarySigmoidLoss():
    def __init__(self, mode='all', focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, epsilon=1e-6):
        super(MILUnarySigmoidLoss, self).__init__()
        self.mode = mode
        self.focal_params = focal_params
        self.epsilon = epsilon
        
    def __call__(self, ypred, mask, gt_boxes):
        loss = mil_unary_sigmoid_loss(ypred, mask, gt_boxes, mode=self.mode,
                                      focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


class MILUnaryApproxSigmoidLoss():
    def __init__(self, mode='all', method='gm', gpower=4, 
                 focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, epsilon=1e-6):
        super(MILUnaryApproxSigmoidLoss, self).__init__()
        self.mode = mode
        self.method = method
        self.gpower = gpower
        self.focal_params = focal_params
        self.epsilon = epsilon 
        
    def __call__(self, ypred, mask, gt_boxes):
        loss = mil_unary_approx_sigmoid_loss(ypred, mask, gt_boxes, mode=self.mode,
                                             method=self.method, gpower=self.gpower, 
                                             focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


class MILUnaryParallelSigmoidLoss():
    def __init__(self, mode='all', angle_params=(-45,46,5), 
                       focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, 
                       obj_size=0, epsilon=1e-6):
        super(MILUnaryParallelSigmoidLoss, self).__init__()
        self.mode           = mode
        self.angle_params   = angle_params
        self.focal_params   = focal_params
        self.obj_size       = obj_size
        self.epsilon        = epsilon
        
    def __call__(self, ypred, mask, crop_boxes):
        loss = mil_parallel_unary_sigmoid_loss(ypred, mask, crop_boxes, angle_params=self.angle_params,
                                               mode=self.mode, focal_params=self.focal_params, 
                                               obj_size=self.obj_size, epsilon=self.epsilon)
        return loss


class MILUnaryParallelApproxSigmoidLoss():
    def __init__(self, mode='all', angle_params=(-45,46,5), method='gm', gpower=4, 
                 focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, 
                 obj_size=0, epsilon=1e-6):
        super(MILUnaryParallelApproxSigmoidLoss, self).__init__()
        self.mode           = mode
        self.angle_params   = angle_params
        self.method         = method
        self.gpower         = gpower
        self.focal_params   = focal_params
        self.obj_size       = obj_size
        self.epsilon        = epsilon
        
    def __call__(self, ypred, mask, crop_boxes):
        loss = mil_parallel_approx_sigmoid_loss(ypred, mask, crop_boxes, angle_params=self.angle_params,
                                               mode=self.mode, focal_params=self.focal_params, 
                                               obj_size=self.obj_size, epsilon=self.epsilon)
        return loss


class MILPairwiseLoss():
    def __init__(self, softmax=True, exp_coef=-1):
        super(MILPairwiseLoss, self).__init__()
        self.softmax = softmax
        self.exp_coef = exp_coef
        
    def __call__(self, ypred, mask):
        loss = mil_pairwise_loss(ypred, mask, softmax=self.softmax, exp_coef=self.exp_coef)
        return loss
