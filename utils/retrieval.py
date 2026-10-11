import json
import os

import numpy as np
import torch

from models.rnn_model import CaptionRNN, ImageEncoder, embed
from utils.data import Vocab, pad_batch, tokenize
from utils.losses import score_matrix


@torch.no_grad()
def encode_images(img_enc, feats, loss_type, device, bs=1024):
    out = []
    for i in range(0, len(feats), bs):
        x = torch.from_numpy(np.asarray(feats[i:i + bs], dtype=np.float32)).to(device)
        out.append(embed(img_enc(x), loss_type).cpu())
    return torch.cat(out)


@torch.no_grad()
def encode_texts(cap_enc, token_lists, vocab, loss_type, device, bs=512):
    out = []
    for i in range(0, len(token_lists), bs):
        caps, lens = pad_batch([vocab.encode(t) for t in token_lists[i:i + bs]])
        out.append(embed(cap_enc(caps.to(device), lens), loss_type).cpu())
    return torch.cat(out)


@torch.no_grad()
def similarity(txt, img, loss_type, device, chunk=50):
    """(Q,D),(N,D) -> (Q,N) numpy similarity; chunked because order-embedding scores are Q*N*D."""
    img = img.to(device)
    rows = [score_matrix(txt[i:i + chunk].to(device), img, loss_type).cpu() for i in range(0, len(txt), chunk)]
    return torch.cat(rows).numpy()


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device)
    cfg = ck["config"]
    cap = CaptionRNN(cfg["vocab_size"], rnn_type=cfg["rnn"], embed_dim=cfg["embed_dim"],
                     hidden_size=cfg["hidden"]).to(device)
    img = ImageEncoder(4096, cfg["embed_dim"]).to(device)
    cap.load_state_dict(ck["caption_encoder"])
    img.load_state_dict(ck["image_encoder"])
    return cap.eval(), img.eval(), cfg


class RetrievalEngine:
    """Text query -> top-k images from a pre-featurised gallery (used by demo.py and the notebook)."""

    def __init__(self, checkpoint, vocab_path, feature_dir, split="test", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cap, self.img, self.cfg = load_checkpoint(checkpoint, self.device)
        self.vocab = Vocab.load(vocab_path)
        self.meta = json.load(open(os.path.join(feature_dir, f"{split}_meta.json")))
        feats = np.load(os.path.join(feature_dir, f"{split}_feats.npy"))
        self.gallery = encode_images(self.img, feats, self.cfg["loss"], self.device)

    def search(self, query, k=5):
        toks = tokenize(query)
        q = encode_texts(self.cap, [toks], self.vocab, self.cfg["loss"], self.device)
        sim = similarity(q, self.gallery, self.cfg["loss"], self.device)[0]
        top = np.argsort(-sim)[:k]
        return [(int(i), float(sim[i]), self.meta[i]) for i in top]
