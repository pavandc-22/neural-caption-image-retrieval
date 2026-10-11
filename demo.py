"""Live demo: type a sentence, get the top-k images.

  python demo.py --query "a guy riding a bike next to a train" --query "dog"      # saves results/demo.png
  python demo.py --gradio                                                         # web UI for the live demo
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from utils.retrieval import RetrievalEngine


def show(engine, queries, img_root, k, out):
    fig, axes = plt.subplots(len(queries), k, figsize=(3 * k, 3.2 * len(queries)), squeeze=False)
    for r, q in enumerate(queries):
        for c, (idx, s, m) in enumerate(engine.search(q, k)):
            ax = axes[r][c]
            ax.imshow(Image.open(os.path.join(img_root, m["filepath"], m["filename"])).convert("RGB"))
            ax.set_title(f"#{c + 1}  ({s:.2f})", fontsize=9); ax.axis("off")
        axes[r][0].text(0, -0.12, q, transform=axes[r][0].transAxes, fontsize=11, weight="bold")
    plt.tight_layout(); plt.savefig(out, dpi=150); print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/gru_glove.pt")
    ap.add_argument("--feature_dir", default="features")
    ap.add_argument("--img_root", default="data/coco/images")
    ap.add_argument("--split", default="test")
    ap.add_argument("--query", action="append")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="results/demo.png")
    ap.add_argument("--gradio", action="store_true")
    a = ap.parse_args()
    eng = RetrievalEngine(a.checkpoint, os.path.join(a.feature_dir, "vocab.json"), a.feature_dir, a.split)

    if a.gradio:
        import gradio as gr

        def run(q, k):
            hits = eng.search(q, int(k))
            return [(os.path.join(a.img_root, m["filepath"], m["filename"]), f"{s:.2f} | {' '.join(m['captions'][0])}")
                    for _, s, m in hits]

        gr.Interface(run, [gr.Textbox(label="Describe the image you want"), gr.Slider(1, 10, 5, step=1, label="top-k")],
                     gr.Gallery(label="Retrieved images"), title="Neural caption-image retrieval").launch()
    else:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        show(eng, a.query or ["a dog", "a guy riding a bike next to a train"], a.img_root, a.k, a.out)


if __name__ == "__main__":
    main()
