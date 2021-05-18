from .vision import VisionDataset
from PIL import Image
import os
import os.path
from typing import Any, Callable, Optional, Tuple
import pandas as pd
import json

class GlaucomaDetection(VisionDataset):
    """`Glaucoma Detection Dataset.

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
            csvFile: str,
            annoFile: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super(GlaucomaDetection, self).__init__(root, transforms, transform, target_transform)
        self.categories = [{'name':'cup','id':0}, {'name':'disc','id':1}] 
        self.csv = pd.read_excel(os.path.join(root,'csv',csvFile))
        # self.csv = self.csv[:50]
        self.ids = list(range(len(self.csv)))
        with open(os.path.join(root,'csv',annoFile), 'r') as f:
            self.annotations = json.load(f)
        self.load_classes()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        img_id = self.ids[index]
        image_name = self.csv['image_name'][img_id]
        annos = self.annotations[image_name]
        # disc = {'area': xx, 'bbox': xx}  
        cup  = {'segmentation': {'x':annos['cup_x'],'y':annos['cup_y']}, 
                'iscrowd':0, 'image_id': img_id, #'image_name': image_name,
               'category_id':0, 'id':img_id*2}
        disc = {'segmentation': {'x':annos['disc_x'],'y':annos['disc_y']}, 
                'iscrowd':0, 'image_id': img_id, #'image_name': image_name,
               'category_id':1, 'id':img_id*2+1}               
        target = [cup, disc]

        img = Image.open(os.path.join(self.root,'images_processing',image_name+'.png')).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.ids)

    def load_classes(self):
        """ Loads the class to label mapping (and inverse) for Glaucoma.
        """
        self.classes             = {}
        self.glaucoma_labels         = {}
        self.glaucoma_labels_inverse = {}
        for c in self.categories:
            self.glaucoma_labels[len(self.classes)] = c['id']
            self.glaucoma_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def num_classes(self):
        """ Number of classes in the dataset. For COCO this is 80.
        """
        return len(self.classes)

    def glaucoma_label_to_label(self, coco_label):
        """ Map COCO label to the label as used in the network.
        COCO has some gaps in the order of labels. The highest label is 90, but there are 80 classes.
        """
        return self.glaucoma_labels_inverse[coco_label]
