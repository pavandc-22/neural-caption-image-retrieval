import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO

class CocoImageCaptionDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        """
        img_dir: Path to the MS COCO image folder
        ann_file: Path to the JSON annotation file
        """
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.anns.keys())
        
        # VGG-19 expects 224x224 inputs. We resize, apply the 10-crop, 
        # and stack them into a single tensor for feature extraction.
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([
                transforms.ToTensor()(crop) for crop in crops
            ]))
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        image = self.transform(image)

        return image, caption

# Example usage to test the loader:
# dataset = CocoImageCaptionDataset(img_dir='data/images', ann_file='data/annotations/captions_train2014.json')
