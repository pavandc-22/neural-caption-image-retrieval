"""Evaluate baseline + every checkpoint on the TEST split (1K-image folds), write tables and figures.

  python evaluate.py --glove data/glove/glove.6B.300d.txt
"""
import argparse
import glob
import json
import os


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.baseline import BaselineLinearRegression
from utils.data import get_glove, get_vocab, load_split
from utils.metrics import evaluate_folds
from utils.retrieval import encode_images, encode_texts, load_checkpoint, similarity


def avg_glove(token_lists, vocab, mat):
    out = np.zeros((len(token_lists), mat.shape[1]), np.float32)
    for i, t in enumerate(token_lists):
        ids = [j for j in vocab.encode(t) if j > 1] or [1]
        out[i] = mat[ids].mean(0)
    return out



def length_curve(ranks, lengths, k=10, min_n=30):
    xs, ys = [], []
    for L in sorted(set(lengths)):
        m = lengths == L
        if m.sum() >= min_n:
            xs.append(L); ys.append((ranks[m] <= k).mean() * 100)
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="features")
    ap.add_argument("--glove", required=True)
    ap.add_argument("--ckpt_glob", default="checkpoints/*.pt")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True)

    vocab = get_vocab(a.feature_dir)
    mat = get_glove(a.feature_dir, a.glove, vocab)
    tr_f, tr_m = load_split(a.feature_dir, "train")
    te_f, te_m = load_split(a.feature_dir, "test")
    n = len(te_m); fold = min(1000, n)
    n_used = (n // fold) * fold
    toks = [c for m in te_m for c in m["captions"]]
    lengths = np.array([len(t) for t in toks]).reshape(n, 5)[:n_used].reshape(-1)
    results, curves = {}, {}

    # --- baselines
    tr_text = avg_glove([c for m in tr_m for c in m["captions"]], vocab, mat)
    base = BaselineLinearRegression(alpha=1.0).fit(tr_text, tr_f.astype(np.float32))
    te_text = avg_glove(toks, vocab, mat)
    te_img = te_f.astype(np.float32)
    r, rk = evaluate_folds(lambda lo, hi: base.similarity(te_text[lo * 5:hi * 5], te_img[lo:hi]), n, fold_size=fold)
    results["Baseline"] = r; curves["Baseline"] = length_curve(rk, lengths)
    avg5 = te_text.reshape(n, 5, -1).mean(1)                 # average of the five captions per image
    r, _ = evaluate_folds(lambda lo, hi: base.similarity(avg5[lo:hi], te_img[lo:hi]), n, fold_size=fold)
    results["Baseline + Weight (5-caption avg)"] = r

    # --- neural models
    for path in sorted(glob.glob(a.ckpt_glob)):
        name = os.path.splitext(os.path.basename(path))[0]
        cap, img, cfg = load_checkpoint(path, dev)
        ie = encode_images(img, te_f, cfg["loss"], dev)
        te = encode_texts(cap, toks, vocab, cfg["loss"], dev)
        r, rk = evaluate_folds(lambda lo, hi: similarity(te[lo * 5:hi * 5], ie[lo:hi], cfg["loss"], dev),
                               n, fold_size=fold)
        results[name] = r; curves[name] = length_curve(rk, lengths)

    json.dump(results, open("results/results.json", "w"), indent=1)
    lines = ["| Method | R@1 | R@5 | R@10 | Mean r | Median r |", "|---|---|---|---|---|---|"]
    for k, v in results.items():
        lines.append(f"| {k} | {v['R@1']:.1f} | {v['R@5']:.1f} | {v['R@10']:.1f} | {v['Mean r']:.1f} | {v['Median r']:.1f} |")
    open("results/results.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # --- figures
    plt.figure(figsize=(5, 3.5))
    for k, (xs, ys) in curves.items():
        plt.plot(xs, ys, "--" if k == "Baseline" else "-", label=k)
    plt.xlabel("Caption length (tokens)"); plt.ylabel("R@10 (%)"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig("results/r10_vs_length.png", dpi=200); plt.close()

    plt.figure(figsize=(5, 3.5))
    for h in sorted(glob.glob("results/history_*.json")):
        hist = json.load(open(h))
        plt.plot([e["epoch"] for e in hist], [e["R@10"] for e in hist], label=h[16:-5])
    if "Baseline + Weight (5-caption avg)" in results:
        plt.axhline(results["Baseline"]["R@10"], ls="--", c="k", lw=1, label="baseline")
    plt.xlabel("Epoch"); plt.ylabel("val R@10 (%)"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=200); plt.close()
    print("saved results/results.{json,md}, r10_vs_length.png, training_curves.png")

# image to vgg to embedding 
if __name__ == "__main__":
    main()
