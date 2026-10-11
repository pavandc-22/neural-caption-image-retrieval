"""Download (1) Karpathy split json, (2) a SUBSET of COCO-2014 images, (3) optionally GloVe.
Images are fetched one by one, so you never need the 19 GB train2014/val2014 zips.

  python scripts/download_data.py --n_train 30000 --n_val 1000 --n_test 5000 --glove
"""
import argparse
import json
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import select_subset

KARPATHY = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
COCO_IMG = "http://images.cocodataset.org/{fp}/{fn}"
GLOVE = "https://nlp.stanford.edu/data/glove.6B.zip"          # 822 MB; mirror: huggingface.co/stanfordnlp/glove


def fetch(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(dst + ".part", "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
        os.replace(dst + ".part", dst)
        return True
    except Exception as e:
        print("FAILED", url, e)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--n_train", type=int, default=30000)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=5000)
    ap.add_argument("--full", action="store_true", help="use the full 113,287/5,000/5,000 split")
    ap.add_argument("--glove", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    coco = os.path.join(a.root, "coco")

    kj = os.path.join(coco, "dataset_coco.json")
    if not os.path.exists(kj):
        z = os.path.join(coco, "caption_datasets.zip")
        print("Downloading Karpathy split ...")
        assert fetch(KARPATHY, z), "could not download Karpathy json - download it manually (see README)"
        with zipfile.ZipFile(z) as zf:
            open(kj, "wb").write(zf.read("dataset_coco.json"))

    n = (None, None, None) if a.full else (a.n_train, a.n_val, a.n_test)
    subset = select_subset(kj, *n)
    json.dump(subset, open(os.path.join(coco, "subset.json"), "w"))
    print({k: len(v) for k, v in subset.items()})

    jobs = [(COCO_IMG.format(fp=im["filepath"], fn=im["filename"]),
             os.path.join(coco, "images", im["filepath"], im["filename"]))
            for s in subset.values() for im in s]
    with ThreadPoolExecutor(a.workers) as ex:
        ok = list(tqdm(ex.map(lambda j: fetch(*j), jobs), total=len(jobs), desc="images"))
    print(f"{sum(ok)}/{len(jobs)} images ready. Re-run the same command to retry failures.")

    if a.glove:
        g = os.path.join(a.root, "glove", "glove.6B.300d.txt")
        if not os.path.exists(g):
            z = os.path.join(a.root, "glove", "glove.6B.zip")
            print("Downloading GloVe (822 MB) ...")
            if fetch(GLOVE, z):
                with zipfile.ZipFile(z) as zf:
                    zf.extract("glove.6B.300d.txt", os.path.dirname(g))
                os.remove(z)


if __name__ == "__main__":
    main()
