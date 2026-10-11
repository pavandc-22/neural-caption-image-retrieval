"""Train the image/caption embedding model on cached VGG features.

  python train.py --run_name gru_glove --rnn gru --glove data/glove/glove.6B.300d.txt
  python train.py --run_name lstm_glove --rnn lstm --glove data/glove/glove.6B.300d.txt
  python train.py --run_name gru_scratch --rnn gru            # no GloVe  (ablation)
  python train.py --run_name gru_order --loss order --glove ...   # order embeddings
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.rnn_model import CaptionRNN, ImageEncoder, embed
from utils.data import FeatureCaptionDataset, collate, get_glove, get_vocab, load_split
from utils.losses import ContrastiveLoss
from utils.metrics import evaluate_folds
from utils.retrieval import encode_images, encode_texts, similarity


def validate(cap, img, feats, meta, vocab, loss_type, dev):
    ie = encode_images(img, feats, loss_type, dev)
    toks = [c for m in meta for c in m["captions"]]            # image-major
    te = encode_texts(cap, toks, vocab, loss_type, dev)
    fold = min(1000, len(meta))
    res, _ = evaluate_folds(lambda lo, hi: similarity(te[lo * 5:hi * 5], ie[lo:hi], loss_type, dev),
                            len(meta), fold_size=fold)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="features")
    ap.add_argument("--run_name", default="gru_glove")
    ap.add_argument("--rnn", choices=["gru", "lstm"], default="gru")
    ap.add_argument("--loss", choices=["cosine", "order"], default="cosine")
    ap.add_argument("--glove", default=None, help="path to glove.6B.300d.txt (omit = train embeddings from scratch)")
    ap.add_argument("--freeze_embeddings", action="store_true")
    ap.add_argument("--embed_dim", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--margin", type=float, default=None, help="default 0.2 (cosine) / 1.0 (order)")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None, help="default 1e-3 (cosine) / 2e-4 (order; larger collapses)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8, help="early-stop after N epochs w/o val improvement")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    margin = a.margin if a.margin is not None else (0.2 if a.loss == "cosine" else 1.0)
    a.lr = a.lr if a.lr is not None else (1e-3 if a.loss == "cosine" else 2e-4)

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    vocab = get_vocab(a.feature_dir)
    glove = get_glove(a.feature_dir, a.glove, vocab) if a.glove else None
    tr_f, tr_m = load_split(a.feature_dir, "train")
    va_f, va_m = load_split(a.feature_dir, "val")
    print(f"train images {len(tr_m)} | val images {len(va_m)} | vocab {len(vocab)} | glove {glove is not None}")

    dl = DataLoader(FeatureCaptionDataset(tr_f, tr_m, vocab), batch_size=a.bs, shuffle=True,
                    collate_fn=collate, drop_last=True, num_workers=0)
    cap = CaptionRNN(len(vocab), hidden_size=a.hidden, embed_dim=a.embed_dim, rnn_type=a.rnn,
                     glove_matrix=glove, freeze_embeddings=a.freeze_embeddings).to(dev)
    img = ImageEncoder(4096, a.embed_dim).to(dev)
    crit = ContrastiveLoss(margin, a.loss)
    params = [p for m in (cap, img) for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr)

    os.makedirs("checkpoints", exist_ok=True); os.makedirs("results", exist_ok=True)
    ck_path = f"checkpoints/{a.run_name}.pt"
    cfg = dict(vars(a), margin=margin, vocab_size=len(vocab), loss=a.loss)
    history, best, bad = [], -1, 0
    for ep in range(1, a.epochs + 1):
        cap.train(); img.train(); t0, tot = time.time(), 0.0
        for f, c, l in dl:
            f, c = f.to(dev), c.to(dev)
            loss = crit(embed(cap(c, l), a.loss), embed(img(f), a.loss))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 2.0)
            opt.step(); tot += loss.item()
        cap.eval(); img.eval()
        v = validate(cap, img, va_f, va_m, vocab, a.loss, dev)
        history.append(dict(epoch=ep, train_loss=tot / len(dl), **v))
        print(f"ep {ep:3d} | loss {tot / len(dl):.4f} | val R@1 {v['R@1']:.1f} R@5 {v['R@5']:.1f} "
              f"R@10 {v['R@10']:.1f} meanr {v['Mean r']:.1f} | {time.time() - t0:.0f}s")
        score = v["R@1"] + v["R@5"] + v["R@10"]
        if score > best:
            best, bad = score, 0
            torch.save(dict(config=cfg, caption_encoder=cap.state_dict(), image_encoder=img.state_dict(),
                            epoch=ep, val=v), ck_path)
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping"); break
        json.dump(history, open(f"results/history_{a.run_name}.json", "w"), indent=1)
    print("best checkpoint ->", ck_path)


if __name__ == "__main__":
    main()
