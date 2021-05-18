import torch
import torch.utils.data
import torchvision_wj

import torchvision_wj.utils.transforms as T
from skimage import draw
import numpy as np
from torchvision_wj.datasets.image import apply_transform, adjust_transform_for_image, TransformParameters
from torchvision_wj.datasets.transform import transform_contour


def convert_poly_to_mask(segmentations, image_shape):
    masks = []
    for polygons in segmentations:
        mask = np.zeros(image_shape, dtype="uint8")
        r1,c1 = draw.polygon(polygons['y'],polygons['x'],shape=image_shape)
        mask[r1,c1] = 1
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, image_shape[0], image_shape[1]), dtype=torch.uint8)
    return masks

def get_box_from_mask(mask):
    ind = np.where(mask>0.5)
    if len(ind[0])==0:
        return [0,0,0,0]
    box = [ind[1].min(), ind[0].min(), ind[1].max(), ind[0].max()] #[x1,y1,x2,y2]
    return box

class ConvertPolysToMask(object):
    def __call__(self, image, target):
        h, w, c = image.shape

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if obj['iscrowd'] == 0]
        
        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_poly_to_mask(segmentations, (h,w))

        boxes = [get_box_from_mask(mask) for mask in masks]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if boxes.shape[1]>0:
            boxes = boxes.reshape(-1, 4)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

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
        area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd
        return image, target

class GlaucomaDetection(torchvision_wj.datasets.GlaucomaDetection):
    def __init__(self, root, csv_file, ann_file, transforms, 
                 transform_generator=None, visual_effect_generator=None,
                 transform_parameters=TransformParameters()):
        super(GlaucomaDetection, self).__init__(root, csv_file, ann_file)
        self._transforms = transforms
        self.transform_generator     = transform_generator
        self.visual_effect_generator = visual_effect_generator
        self.transform_parameters    = transform_parameters

    def __getitem__(self, idx):
        img, target = super(GlaucomaDetection, self).__getitem__(idx)
        # img = np.ascontiguousarray(img)
        img = np.array(img, copy=True)
        # print(img.flags['WRITEABLE'])
        
        if self.transform_generator is not None:
            transform = adjust_transform_for_image(next(self.transform_generator), img, 
                                                   self.transform_parameters.relative_translation)
            img = apply_transform(transform, img, self.transform_parameters)
            for t in target:
                x,y = transform_contour(transform, t['segmentation']['x'], t['segmentation']['y'])
                t['segmentation']['x'] = x
                t['segmentation']['y'] = y

        if self.visual_effect_generator is not None:
            visual_effect = next(self.visual_effect_generator)
            img = visual_effect(img)
        
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target
    


def get_glaucoma(root, csv_file, transforms, mode='instances', 
                 transform_generator=None, visual_effect_generator=None):

    t = [ConvertPolysToMask()]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    anno_file = 'annotation_{}.json'.format(csv_file)
    dataset = GlaucomaDetection(root, csv_file+'.xlsx', anno_file, transforms=transforms,
                                transform_generator=transform_generator,
                                visual_effect_generator=visual_effect_generator)

    # if "train" in csv_file:
    #     dataset = _coco_remove_images_without_annotations(dataset)

    return dataset
