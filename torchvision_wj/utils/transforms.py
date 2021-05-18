import random
import torch
import copy
from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Normalizer(object):
    def __init__(self, mode='zscore', mean=None, std=None):
        self.mode = mode
        if self.mode=='customize':
            assert mean is not None and std is not None
            self.mean = mean
            self.std  = std

    def __call__(self, image, target):
        if self.mode=='zscore':
            self.mean = torch.mean(image)
            self.std  = torch.std(image)
        image = (image-self.mean)/self.std
        return image, target


class RandomCrop(object):
    def __init__(self, prob, crop_size, weak=False):
        self.prob = prob # prob to crop from object region
        self.crop_size = crop_size
        self.weak = weak

    def __call__(self, image, target):
        if self.weak:
            mask = image.new_full(image.shape[1:],0,device=image.device)
            for box in target['boxes']:
                box = box.int()
                mask[box[1]:box[3]+1,box[0]:box[2]+1] = 1
            mask = mask > 0.5
        else:
            mask = torch.sum(target['masks'],dim=0)>0.5
        crop_loc = self.crop_region(mask)
        image = image[:,crop_loc[0]:crop_loc[1],crop_loc[2]:crop_loc[3]]
        masks = target['masks'][:,crop_loc[0]:crop_loc[1],crop_loc[2]:crop_loc[3]]
        target['masks'] = masks
        bbox  = target['boxes']
        label = target['labels']
        assert bbox.shape[0]==len(label), '#boxes = {}, #labels = {} in RandomCrop'.format(bbox.shape[0],len(label))
  
        bbox[:,0::2] = bbox[:,0::2] - crop_loc[2]
        bbox[:,1::2] = bbox[:,1::2] - crop_loc[0]
        target['boxes_org'] = copy.deepcopy(bbox)
        h, w = image.shape[-2:]
        bbox[:,0::2] = torch.clamp(bbox[:,0::2], 0, w)
        bbox[:,1::2] = torch.clamp(bbox[:,1::2], 0, h)
        flag = (bbox[:,2]-bbox[:,0]>5)&(bbox[:,3]-bbox[:,1]>5) # 2 is set such that objects with size less than 5x5 are not considered
        target['boxes'] = bbox[flag,:]
        target['labels'] = label[flag]
        return image, target

    def crop_region(self, mask):
        if random.random() < self.prob: 
            bw = mask ## crop from object
        else: 
            bw = mask==False ## crop from background
        xl,yl = self.crop_size[0]//2, self.crop_size[1]//2
        xh,yh = self.crop_size[0]-xl, self.crop_size[1]-yl
        ind = torch.nonzero(bw, as_tuple=True)
        if len(ind[0])==0:
            x,y = bw.shape[0]//2, bw.shape[1]//2
        else:
            loc = torch.randint(high=len(ind[0]),size=(1,))
            x,y = ind[0][loc],ind[1][loc]
        xmin,ymin = x-xl,y-yl
        xmax,ymax = x+xh,y+yh
        if xmin<0:
            xmin,xmax = 0, self.crop_size[0]
        if ymin<0:
            ymin,ymax = 0, self.crop_size[1]
        if xmax>=bw.shape[0]:
            xmin,xmax = bw.shape[0]-self.crop_size[0], bw.shape[0]
        if ymax>=bw.shape[1]:
            ymin,ymax = bw.shape[1]-self.crop_size[1], bw.shape[1]
        return [xmin,xmax,ymin,ymax]

def resize_keypoints(keypoints, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=keypoints.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=keypoints.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_h, ratio_w = ratios
    resized_data = keypoints.clone()
    if torch._C._get_tracing_state():
        resized_data_0 = resized_data[:, :, 0] * ratio_w
        resized_data_1 = resized_data[:, :, 1] * ratio_h
        resized_data = torch.stack((resized_data_0, resized_data_1, resized_data[:, :, 2]), dim=2)
    else:
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h
    return resized_data

def resize_boxes(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

def _resize_image_and_masks(image, scale_factor, target):
    # type: (Tensor, float, float, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
    
    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode='bilinear', #recompute_scale_factor=True,
        align_corners=False)[0]

    if target is None:
        return image, target

    if "masks" in target:
        mask = target["masks"]
        mask = torch.nn.functional.interpolate(mask[:, None].float(), scale_factor=scale_factor)[:, 0].byte()
        target["masks"] = mask
    return image, target

class Resize(object):
    def __init__(self, width=None, min_size=None, max_size=None):
        self.width = width
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        # type: (Tensor, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
        h, w = image.shape[-2:]
        if (self.min_size is not None) & (self.max_size is not None):
            assert self.width is None
            if self.training:
                self_min_size = float(self.torch_choice(self.min_size))
            else:
                # FIXME assume for now that testing uses the largest scale
                self_min_size = float(self.min_size[-1])
            self_max_size = float(self.max_size)
            im_shape = torch.tensor(image.shape[-2:])
            min_size = float(torch.min(im_shape))
            max_size = float(torch.max(im_shape))
            scale_factor = self_min_size / min_size
            if max_size * scale_factor > self_max_size:
                scale_factor = self_max_size / max_size
        elif self.width is not None:
            scale_factor = self.width / w
        image, target = _resize_image_and_masks(image, scale_factor, target)

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox
        target["boxes_org"] = bbox

        label = target['labels']
        assert bbox.shape[0]==len(label), '#boxes = {}, #labels = {} in Resize'.format(bbox.shape[0],len(label))

        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
            target["keypoints"] = keypoints
        
        return image, target

    def torch_choice(self, k):
        # type: (List[int]) -> int
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]