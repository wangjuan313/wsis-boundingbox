import torch
import torch.utils.data
import torchvision_wj

import torchvision_wj.utils.transforms as T
import numpy as np
from torchvision_wj.datasets.image import apply_transform, adjust_transform_for_image, TransformParameters
from torchvision_wj.datasets.transform import transform_contour

def get_box_from_mask(mask):
    ind = np.where(mask>0.5)
    if len(ind[0])==0:
        return [0,0,0,0]
    box = [ind[1].min(), ind[0].min(), ind[1].max(), ind[0].max()] #[x1,y1,x2,y2]
    return box

class ConvertToTensor(object):
    def __call__(self, image, target):
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        masks = torch.tensor(target['masks'], dtype=torch.uint8)
        masks = masks[None,:,:]
        boxes = torch.tensor(target['boxes'], dtype=torch.float32)

        classes = [target["category_id"]]*boxes.shape[0]
        classes = torch.tensor(classes, dtype=torch.int64)

        keypoints = None
        if "keypoints" in target.keys():
            keypoints = torch.as_tensor(target['keypoints'], dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        iscrowd = torch.tensor(target["iscrowd"])

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id
        target['boxes_org'] = boxes
        if keypoints is not None:
            target["keypoints"] = keypoints

        assert boxes.shape[0]==len(classes)

        # for conversion to coco api
        target["area"] = (boxes[:,2]-boxes[:,0]+1)*(boxes[:,3]-boxes[:,1]+1)
        target["iscrowd"] = iscrowd 
        return image, target

class PromiseDetection(torchvision_wj.datasets.PromiseDetection):
    def __init__(self, root, image_folder, gt_folder, transforms, margin=0,
                 transform_generator=None, visual_effect_generator=None,
                 transform_parameters=TransformParameters()):
        super(PromiseDetection, self).__init__(root, image_folder, gt_folder, margin)
        self._transforms = transforms
        self.transform_generator = transform_generator
        self.visual_effect_generator = visual_effect_generator
        self.transform_parameters = transform_parameters

    def __getitem__(self, idx):
        img, target = super(PromiseDetection, self).__getitem__(idx)
        # img = np.ascontiguousarray(img)
        img = np.array(img, copy=True)
        # print(img.flags['WRITEABLE'])
        
        if self.transform_generator is not None:
            transform = adjust_transform_for_image(next(self.transform_generator), img, 
                                                   self.transform_parameters.relative_translation)
            img = apply_transform(transform, img, self.transform_parameters)
            # for t in target:
            #     x,y = transform_contour(transform, t['segmentation']['x'], t['segmentation']['y'])
            #     t['segmentation']['x'] = x
            #     t['segmentation']['y'] = y

        if self.visual_effect_generator is not None:
            visual_effect = next(self.visual_effect_generator)
            img = visual_effect(img)
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def get_promise(root, image_folder, gt_folder, transforms, margin=0,
                transform_generator=None, visual_effect_generator=None):

    t = [ConvertToTensor()]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    dataset = PromiseDetection(root, image_folder, gt_folder, margin=margin,
                               transforms=transforms,
                               transform_generator=transform_generator,
                               visual_effect_generator=visual_effect_generator)

    return dataset
