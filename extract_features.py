"""Run frozen VGG-19 once over every selected image and cache fc7 features (fp16).
Ten-crop averaging as in the report; --crop center is ~10x faster.

  python extract_features.py --crop ten
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from models.extractors import VGG19Extractor

NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # ImageNet stats (required by VGG weights)


def stack_crops(crops):
    """Module-level (picklable) so DataLoader workers also work on Windows."""
    return torch.stack([NORM(T.ToTensor()(c)) for c in crops])


class ImageFolderList(Dataset):
    def __init__(self, items, root, crop):
        self.items, self.root = items, root
        if crop == "ten":
            self.tf = T.Compose([T.Resize((256, 256)), T.TenCrop(224),
                                 T.Lambda(stack_crops)])
        else:
            self.tf = T.Compose([T.Resize((256, 256)), T.CenterCrop(224), T.ToTensor(), NORM])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        im = self.items[i]
        img = Image.open(os.path.join(self.root, im["filepath"], im["filename"])).convert("RGB")
        return self.tf(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="data/coco/subset.json")
    ap.add_argument("--img_root", default="data/coco/images")
    ap.add_argument("--out", default="features")
    ap.add_argument("--crop", choices=["ten", "center"], default="ten")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_pretrained", action="store_true", help="random weights (smoke tests only)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    net = VGG19Extractor(pretrained=not a.no_pretrained).to(dev)
    subset = json.load(open(a.subset))
    os.makedirs(a.out, exist_ok=True)
    for split, items in subset.items():
        dst = os.path.join(a.out, f"{split}_feats.npy")
        if os.path.exists(dst):
            print(split, "already extracted, skipping")
            continue
        dl = DataLoader(ImageFolderList(items, a.img_root, a.crop), batch_size=a.bs,
                        num_workers=a.workers, shuffle=False)
        feats = []
        for x in tqdm(dl, desc=split):
            with torch.autocast(dev, enabled=(dev == "cuda")):
                f = net(x.to(dev))
            feats.append(f.float().cpu().numpy().astype(np.float16))
        np.save(dst, np.concatenate(feats))
        json.dump(items, open(os.path.join(a.out, f"{split}_meta.json"), "w"))
        print(split, "->", dst)


if __name__ == "__main__":
    main()
