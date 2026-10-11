"""Data utilities: Karpathy-split subset selection, vocabulary, GloVe, feature datasets."""
import json
import os
import random
import re
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

PAD, UNK = 0, 1


# ---------------------------------------------------------------- selection
def select_subset(karpathy_json, n_train, n_val, n_test, seed=0):
    """Pick images from the Karpathy COCO split. 'restval' is part of train (113,287 train imgs).
    Returns {'train': [...], 'val': [...], 'test': [...]} with per-image dicts
    {filepath, filename, captions: [token lists]}. Pass n=None for the full split."""
    data = json.load(open(karpathy_json))["images"]
    by = {"train": [], "val": [], "test": []}
    for im in data:
        s = "train" if im["split"] in ("train", "restval") else im["split"]
        caps = [x["tokens"] for x in im["sentences"]][:5]
        if len(caps) == 5:
            by[s].append({"filepath": im["filepath"], "filename": im["filename"], "captions": caps})
    rng = random.Random(seed)
    out = {}
    for s, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        items = by[s]
        if n is not None and n < len(items):
            rng.shuffle(items)
            items = items[:n]
        out[s] = items
    return out


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------- vocab / glove
class Vocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    @classmethod
    def build(cls, caption_lists, min_count=5):
        c = Counter(w for caps in caption_lists for cap in caps for w in cap)
        return cls(["<pad>", "<unk>"] + sorted(w for w, n in c.items() if n >= min_count))

    def encode(self, tokens):
        return [self.stoi.get(w, UNK) for w in tokens]

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        json.dump(self.itos, open(path, "w"))

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path)))


def load_glove_matrix(glove_path, vocab, dim=300, seed=0):
    """(V, dim) matrix: GloVe vector if the word is known, small random vector otherwise, zeros for <pad>."""
    rng = np.random.RandomState(seed)
    mat = rng.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    mat[PAD] = 0
    found = 0
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            w, rest = line.split(" ", 1)
            i = vocab.stoi.get(w)
            if i is not None and i > UNK:
                mat[i] = np.asarray(rest.split(), dtype=np.float32)
                found += 1
    print(f"GloVe covers {found}/{len(vocab) - 2} vocabulary words")
    return mat


def pad_batch(seqs):
    lengths = torch.tensor([max(len(s), 1) for s in seqs])
    out = torch.zeros(len(seqs), int(lengths.max()), dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s if s else [UNK])
    return out, lengths


# ---------------------------------------------------------------- datasets
class FeatureCaptionDataset(Dataset):
    """One item per IMAGE: (fc7 feature, one random caption of its five). Unique images per batch
    means no false negatives in the in-batch contrastive loss."""

    def __init__(self, feats, meta, vocab):
        self.feats = feats
        self.caps = [[vocab.encode(c) for c in m["captions"]] for m in meta]

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return torch.from_numpy(self.feats[i].astype(np.float32)), random.choice(self.caps[i])


def collate(batch):
    feats = torch.stack([b[0] for b in batch])
    caps, lens = pad_batch([b[1] for b in batch])
    return feats, caps, lens


def load_split(feature_dir, split):
    feats = np.load(os.path.join(feature_dir, f"{split}_feats.npy"))
    meta = json.load(open(os.path.join(feature_dir, f"{split}_meta.json")))
    return feats, meta


def get_vocab(feature_dir, min_count=5):
    """Build (or load cached) vocabulary from the TRAIN captions only."""
    path = os.path.join(feature_dir, "vocab.json")
    if os.path.exists(path):
        return Vocab.load(path)
    meta = json.load(open(os.path.join(feature_dir, "train_meta.json")))
    vocab = Vocab.build([m["captions"] for m in meta], min_count)
    vocab.save(path)
    return vocab


def get_glove(feature_dir, glove_path, vocab):
    cache = os.path.join(feature_dir, "glove_matrix.npy")
    if os.path.exists(cache):
        return np.load(cache)
    mat = load_glove_matrix(glove_path, vocab)
    np.save(cache, mat)
    return mat
