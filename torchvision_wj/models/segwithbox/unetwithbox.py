import copy
import torch
from torch import nn
from torchvision_wj.models.batch_image import BatchImage
from torchvision_wj.utils.losses import *
from torch.jit.annotations import Tuple, List, Dict, Optional


__all__ = ['UNetWithBox']

class UNetWithBox(nn.Module):
    """
    Implements UNetWithBox.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.
    """
    
    def __init__(self, model, losses, loss_weights, softmax, batch_image=BatchImage,
                size_divisible=32):

        super(UNetWithBox, self).__init__()        
        self.model = model
        nb_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('##### trainable parameters in model: {}'.format(nb_params))
        self.losses_func = losses
        self.loss_weights = loss_weights
        self.softmax = softmax
        self.batch_image = batch_image(size_divisible)


    def sigmoid_compute_seg_loss(self, seg_preds, targets, image_shape):
        device = seg_preds[0].device
        dtype  = seg_preds[0].dtype
        all_labels = [t['labels'] for t in targets]
        ytrue = torch.stack([t['masks'] for t in targets],dim=0).long()
        label_unique = torch.unique(torch.cat(all_labels, dim=0))
        seg_losses = {}
        for nb_level in range(len(seg_preds)):
            preds = seg_preds[nb_level]
            stride = image_shape[-1]/preds.shape[-1]

            # masks = preds.new_full(preds.shape,1,device=device)
            # ## compute masks based on bounding box
            # gt_boxes_org = []
            # for n_img, target in enumerate(targets):
            #     if target['boxes_org'].shape[0]==0:
            #         continue
            #     bb = torch.round(target['boxes_org']/stride).type(torch.int32)
            #     ext = torch.tensor([[n_img,0]], device=device)
            #     gt_boxes_org.append(torch.cat([ext,bb], dim=1))
            # gt_boxes_org = torch.cat(gt_boxes_org, dim=0)
            # h, w = preds.shape[-2:]
            # bbox  = copy.deepcopy(gt_boxes_org[:,2:])
            # bbox[:,0::2] = torch.clamp(bbox[:,0::2], 0, w)
            # bbox[:,1::2] = torch.clamp(bbox[:,1::2], 0, h)
            # flag = (bbox[:,2]-bbox[:,0]>5)&(bbox[:,3]-bbox[:,1]>5)
            # gt_boxes_org = gt_boxes_org[flag,:]

            mask = preds.new_full(preds.shape,0,device=preds.device)
            crop_boxes = []
            gt_boxes   = []
            for n_img, target in enumerate(targets):
                boxes = torch.round(target['boxes']/stride).type(torch.int32)
                labels = target['labels']
                for n in range(len(labels)):
                    box = boxes[n,:]
                    c   = labels[n]#.item()
                    mask[n_img,c,box[1]:box[3]+1,box[0]:box[2]+1] = 1

                    height, width = (box[2]-box[0]+1)/2.0, (box[3]-box[1]+1)/2.0
                    r  = torch.sqrt(height**2+width**2)
                    cx = (box[2]+box[0]+1)//2
                    cy = (box[3]+box[1]+1)//2
                    # print('//// box ////',box, cx, cy, r)
                    crop_boxes.append(torch.tensor([n_img, c, cx, cy, r]))
                    gt_boxes.append(torch.tensor([n_img, c, box[0], box[1], box[2], box[3]], dtype=torch.int32, device=device))
            if len(crop_boxes)==0:
                crop_boxes = torch.empty((0,5), device=device)
            else:
                crop_boxes = torch.stack(crop_boxes, dim=0)
            if len(gt_boxes)==0:
                gt_boxes = torch.empty((0,6), device=device)
            else:
                gt_boxes = torch.stack(gt_boxes, dim=0) 

            # print('#boxes',crop_boxes.shape[0],gt_boxes.shape[0])
            assert crop_boxes.shape[0]==gt_boxes.shape[0]

            kwargs_opt = {'ypred':preds, 'ytrue':ytrue, 'mask':mask, 'gt_boxes':gt_boxes, 'crop_boxes':crop_boxes}
            for loss_func, loss_w in zip(self.losses_func,self.loss_weights):
                loss_keys = loss_func.__call__.__code__.co_varnames
                loss_params = {key:kwargs_opt[key] for key in kwargs_opt.keys() if key in loss_keys}
                loss_v = loss_func(**loss_params)*loss_w

                key_prefix = type(loss_func).__name__+'/'+str(nb_level)+'/'
                loss_v = {key_prefix+str(n):loss_v[n] for n in range(len(loss_v))}
                seg_losses.update(loss_v)
        
            # from imageio import imwrite
            # import numpy as np
            # for k in range(ytrue.shape[0]):
            #     true = ytrue[k,0].cpu().numpy().astype(float)
            #     msk  = mask[k,0].cpu().numpy()
            #     print(np.unique(true),np.unique(msk))
            #     concat = true+msk
            #     concat = 255*(concat-concat.min())/(concat.max()-concat.min()+1e-6)
            #     concat = concat.astype(np.uint8)
            #     # concat = (np.hstack([true,msk])*255).astype(np.uint8)
            #     imwrite(str(k)+'.png', concat)
            # sys.exit()
        return seg_losses

    def softmax_compute_seg_loss(self, seg_preds, targets, image_shape, eps=1e-6):
        device = seg_preds[0].device
        dtype  = seg_preds[0].dtype
        all_labels = [t['labels'] for t in targets]
        ytrue = torch.stack([t['masks'] for t in targets],dim=0).long()
        label_unique = torch.unique(torch.cat(all_labels, dim=0))
        seg_losses = {}
        for nb_level in range(len(seg_preds)):
            preds = seg_preds[nb_level]
            stride = image_shape[-1]/preds.shape[-1]

            mask = preds.new_full(preds.shape,0,device=preds.device)
            crop_boxes = []
            gt_boxes   = []
            for n_img, target in enumerate(targets):
                boxes = torch.round(target['boxes']/stride).type(torch.int32)
                labels = target['labels']
                for n in range(len(labels)):
                    box = boxes[n,:]
                    c   = labels[n]#.item()
                    mask[n_img,c,box[1]:box[3]+1,box[0]:box[2]+1] = 1

                    height, width = (box[2]-box[0]+1)/2.0, (box[3]-box[1]+1)/2.0
                    r  = torch.sqrt(height**2+width**2)
                    cx = (box[2]+box[0]+1)//2
                    cy = (box[3]+box[1]+1)//2
                    # print('//// box ////',box, cx, cy, r)
                    crop_boxes.append(torch.tensor([n_img, c, cx, cy, r]))
                    gt_boxes.append(torch.tensor([n_img, c, box[0], box[1], box[2], box[3]], dtype=torch.int32, device=device))
            if len(crop_boxes)==0:
                crop_boxes = torch.empty((0,5), device=device)
            else:
                crop_boxes = torch.stack(crop_boxes, dim=0)
            if len(gt_boxes)==0:
                gt_boxes = torch.empty((0,6), device=device)
            else:
                gt_boxes = torch.stack(gt_boxes, dim=0) 

            # print('#boxes',crop_boxes.shape[0],gt_boxes.shape[0])
            assert crop_boxes.shape[0]==gt_boxes.shape[0]

            kwargs_opt = {'ypred':preds, 'ytrue':ytrue, 'mask':mask, 'gt_boxes':gt_boxes, 'crop_boxes':crop_boxes}
            for loss_func, loss_w in zip(self.losses_func,self.loss_weights):
                loss_keys = loss_func.__call__.__code__.co_varnames
                loss_params = {key:kwargs_opt[key] for key in kwargs_opt.keys() if key in loss_keys}
                loss_v = loss_func(**loss_params)*loss_w

                key_prefix = type(loss_func).__name__+'/'+str(nb_level)+'/'
                loss_v = {key_prefix+str(n):loss_v[n] for n in range(len(loss_v))}
                seg_losses.update(loss_v)

        return seg_losses
    
    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.batch_image(images, targets)
        if torch.isnan(images.tensors).sum()>0:
            print('image is nan ..............')
        if torch.isinf(images.tensors).sum()>0:
            print('image is inf ..............')   

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        seg_preds = self.model(images.tensors)
        # print('--------------image and seg outputs: ')
        # print(images.tensors.shape, seg_preds[0].shape)
        # for n_img, target in enumerate(targets):
        #     print(n_img, target['masks'].shape)

        ## calculate losses
        assert targets is not None
        if self.softmax:
            losses = self.softmax_compute_seg_loss(seg_preds, targets, images.tensors.shape)
        else:
            losses = self.sigmoid_compute_seg_loss(seg_preds, targets, images.tensors.shape)

        return losses, seg_preds

