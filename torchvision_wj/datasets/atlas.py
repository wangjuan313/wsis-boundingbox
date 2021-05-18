from .vision import VisionDataset
from PIL import Image
import os
import io
from typing import Any, Callable, Optional, Tuple
from skimage import measure
import numpy as np
from skimage.morphology import remove_small_objects

class AtlasDetection(VisionDataset):
    """`Atlas Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
            self,
            root: str,
            image_folder: str,
            gt_folder: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super(AtlasDetection, self).__init__(transform, target_transform, transforms)
        self.categories = [{'name':'lesion','id':0}]         
        self.image_folder = os.path.join(root,image_folder)
        self.gt_folder    = os.path.join(root,gt_folder)
        self.image_names  = os.listdir(self.image_folder)
        assert len(self.image_names) == len(os.listdir(self.gt_folder))
        self.ids    = list(range(len(self.image_names)))
        
        self.images = [] #self.load_images(image_folder, in_memory=True)
        self.gt     = [] #self.load_images(gt_folder, in_memory=True)
        self.load_classes()
        
    def load_images(self, folder, in_memory: bool, quiet=False):
        def load(folder, filename):
            p = os.path.join(folder, filename)
            if in_memory:
                with open(p, 'rb') as data:
                    res = io.BytesIO(data.read())
                return res
            return p
        if in_memory and not quiet:
            print("> Loading the data in memory...")

        files = [load(folder, im) for im in self.image_names] 

        return files

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        # img = np.array(Image.open(self.images[index]), copy=True)
        # gt  = np.array(Image.open(self.gt[index]), copy=True)>0
        img = np.array(Image.open(os.path.join(self.image_folder, self.image_names[index])), copy=True)
        gt = np.array(Image.open(os.path.join(self.gt_folder, self.image_names[index])), copy=True)>0
        img_id = self.ids[index]
        obj_seg, bbox = self.binary2boxcoords(remove_small_objects(gt,6))
        
        target  = {'masks': gt, 'iscrowd': 0, 
                   'category_id': 0, 'id': img_id, 'image_id':img_id}  
        if len(bbox)>0:
            target['boxes'] = np.vstack(bbox)
            # print(index, gt.dtype, self.image_names[index], np.sum(gt), target['boxes'])
        else:
            target['boxes'] = np.empty((0,4))

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    @staticmethod
    def binary2boxcoords(seg):
        assert set(np.unique(seg)).issubset([0, 1])
        assert len(seg.shape)==2  # ensure the 2d shape
        
        blobs, n_blob = measure.label(seg, background=0, return_num=True)            
        assert set(np.unique(blobs)) <= set(range(0, n_blob + 1)), np.unique(blobs)
    
        obj_coords = []
        obj_seg = []
        for b in range(1, n_blob + 1):
            blob_mask = blobs == b
            obj_seg.append(blob_mask)
    
            assert blob_mask.dtype == np.bool, blob_mask.dtype
            # assert set(np.unique(blob_mask)) == set([0, 1])
    
            coords = np.argwhere(blob_mask)
    
            x1, y1 = coords.min(axis=0)
            x2, y2 = coords.max(axis=0)

            if (x1<x2)&(y1<y2):
                obj_coords.append([y1, x1, y2, x2])
        # assert len(obj_coords) == n_blob
        
        # if len(obj_coords)==0:
        #     obj_coords.append([0,0,-1,-1])
    
        return obj_seg, obj_coords

    def __len__(self) -> int:
        return len(self.image_names)

    def load_classes(self):
        """ Loads the class to label mapping (and inverse) for Glaucoma.
        """
        self.classes                = {}
        self.atlas_labels         = {}
        self.atlas_labels_inverse = {}
        for c in self.categories:
            self.atlas_labels[len(self.classes)] = c['id']
            self.atlas_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def num_classes(self):
        """ Number of classes in the dataset. For COCO this is 80.
        """
        return len(self.classes)

    def atlas_label_to_label(self, label):
        """ Map COCO label to the label as used in the network.
        COCO has some gaps in the order of labels. The highest label is 90, but there are 80 classes.
        """
        return self.atlas_labels_inverse[label]
