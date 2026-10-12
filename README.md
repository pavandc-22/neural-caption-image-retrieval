# Neural Caption-Image Retrieval (UE24CS352A ML Mini-Project)

Given a text description, retrieve the matching images from a gallery.
Images -> frozen **VGG-19 fc7** (10-crop average). Captions -> **GloVe-initialised GRU/LSTM**. Both are mapped to a shared
1024-d space and trained with a bidirectional margin ranking loss (in-batch negatives). Compared against a
ridge-regression baseline (averaged GloVe -> VGG space). Metric: Recall@K on 1K-image galleries (COCO, Karpathy split).

Based on: Vendrov et al., *Order-Embeddings of Images and Language* (2015); Qian & Lamberti, *Neural Caption-Image Retrieval* (CS229).

## Layout
```
scripts/download_data.py   Karpathy split + COCO image subset (+ GloVe)
extract_features.py        VGG-19 fc7 features, cached to features/
train.py                   GRU / LSTM / order-embedding training, early stopping on val
evaluate.py                baseline + all checkpoints on test -> results/*.md|json|png
demo.py, notebooks/demo.ipynb   text query -> top-k images (CLI, Gradio web UI, notebook)
models/  utils/            encoders, baseline, losses, metrics, data utils
```

## Setup (a GPU - e.g. free Colab/Kaggle T4 - is strongly recommended for step 2)
```bash
pip install -r requirements.txt

# 1. data (images are downloaded one-by-one: no 19 GB zips). ~35k images by default.
python scripts/download_data.py --n_train 30000 --n_val 1000 --n_test 5000 --glove
#    full split: add --full (123k images, ~19 GB)

# 2. cache image features once (--crop center is ~10x faster, slightly lower accuracy)
python extract_features.py --crop ten

# 3. train (each run is minutes: features are cached)
python train.py --run_name gru_glove   --rnn gru  --glove data/glove/glove.6B.300d.txt
python train.py --run_name lstm_glove  --rnn lstm --glove data/glove/glove.6B.300d.txt
python train.py --run_name gru_scratch --rnn gru                       # ablation: no GloVe
python train.py --run_name gru_order   --rnn gru --loss order --glove data/glove/glove.6B.300d.txt

# 4. evaluate on the test split
python evaluate.py --glove data/glove/glove.6B.300d.txt

# 5. demo
python demo.py --query "a guy riding a bike next to a train" --query "dog"
python demo.py --gradio
```

## Evaluation protocol
Test images are split into folds of 1,000. Every one of the 5 captions of a fold's images is a query against that fold's
1,000 images; R@K / mean rank are averaged over folds. Ties are counted against the model.
"Baseline + Weight" uses the average of the five caption vectors of an image as a single query (as in the report).

## Results
Setting actually used for the reported numbers: **10,000 train / 500 val / 1,000 test images** (random subset of the
Karpathy split), VGG-19 fc7 with **centre-crop** features, up to **20 epochs** with early stopping, seed 0.
Test set = 1,000 images, each of the 5,000 captions is a query (R@K in %, ties counted against the model).

| Method | R@1 | R@5 | R@10 | Mean r | Median r |
|---|---|---|---|---|---|
| Baseline (avg. GloVe + ridge) | 3.6 | 12.9 | 20.3 | 187.9 | 73.0 |
| Baseline + Weight (5-caption avg.) | 3.7 | 14.0 | 22.0 | 177.1 | 66.0 |
| Order embedding + GloVe (GRU) | 8.4 | 30.0 | 46.5 | 29.4 | 12.0 |
| GRU, no GloVe | 17.9 | 46.6 | 64.1 | 23.4 | 6.0 |
| LSTM + GloVe | 18.2 | 49.7 | 67.4 | 19.6 | 6.0 |
| **GRU + GloVe** | **19.9** | **51.1** | **68.2** | 19.8 | 5.0 |

Figures: `results/training_curves.png`, `results/r10_vs_length.png`, `results/demo.png`. Raw numbers: `results/results.json`,
per-epoch logs: `results/history_*.json`.

Commands used:
```bash
python scripts/download_data.py --n_train 10000 --n_val 500 --n_test 1000 --glove
python extract_features.py --crop center --bs 32
python train.py --run_name gru_glove --rnn gru --glove data/glove/glove.6B.300d.txt --epochs 20   # + lstm_glove, gru_scratch, gru_order (--loss order)
python evaluate.py --glove data/glove/glove.6B.300d.txt
```
Absolute numbers are lower than the original report (about 86% R@10 on the full training set) because of the smaller training subset and single-crop features.
Trained checkpoints and cached features are not stored in git (see `.gitignore`); re-create them with the commands above.
