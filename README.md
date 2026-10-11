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
Fill in from `results/results.md` after running `evaluate.py`.
