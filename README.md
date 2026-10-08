# PaviaU 1D CNN for Hyperspectral Image Classification

A clean, reproducible implementation of a **1D Convolutional Neural Network (1D CNN)** for pixel-wise classification of the **Pavia University (PaviaU)** hyperspectral dataset.

The model treats the spectral signature of each pixel as a one-dimensional sequence and predicts its land-cover class. The repository is organized so that the experiment can be reproduced from a script while the accompanying notebook provides an interactive research workflow.

## Project structure

```text
PaviaU-1D-CNN/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   └── README.md
├── notebooks/
│   └── PaviaU_1D_CNN.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── models/
│   └── .gitkeep
└── results/
    ├── figures/
    └── metrics/
```

## Methodology

The current experiment follows the methodology implemented in the supplied notebook:

1. Load `PaviaU.mat` and `PaviaU_gt.mat`.
2. Reshape the hyperspectral cube into one spectral vector per pixel.
3. Use **20% of the labeled pixels from each class for training** and the remaining labeled pixels for validation.
4. Normalize spectral values by dividing by `255`.
5. Reshape each spectral signature to `(bands, 1)` for 1D convolution.
6. Train a two-block 1D CNN with dropout and max pooling.
7. Predict every pixel to generate a classification map.
8. Evaluate the labeled pixels using overall accuracy, Cohen's kappa, per-class accuracy, and a confusion matrix.

### CNN architecture

```text
Input: spectral signature
        ↓
Conv1D (32 filters, kernel size 5, ReLU)
        ↓
Dropout (0.30)
        ↓
MaxPooling1D (pool size 2)
        ↓
Conv1D (32 filters, kernel size 5, ReLU)
        ↓
Dropout (0.30)
        ↓
MaxPooling1D (pool size 2)
        ↓
Flatten
        ↓
Dense (128, ReLU)
        ↓
Dense (number of classes, Softmax)
```

## Dataset

The code expects the following MATLAB files:

```text
data/PaviaU.mat
data/PaviaU_gt.mat
```

The supplied notebook loads the MATLAB variables `paviaU` and `paviaU_gt` from these files.

**Dataset files are intentionally not included in this initial repository package.** Before publishing the repository, confirm the dataset's redistribution/license terms. If redistribution is permitted, place the two `.mat` files under `data/`.

See [`data/README.md`](data/README.md) for the expected files and setup instructions.

## Installation

Python 3.10/3.11 is recommended for a stable TensorFlow environment.

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd PaviaU-1D-CNN
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the experiment

After placing the dataset files in `data/`, run:

```bash
python -m src.train
```

The script trains the model, generates the full-scene classification map, and writes outputs under `results/`.

## Notebook

The cleaned research notebook is available at:

`notebooks/PaviaU_1D_CNN.ipynb`

It uses repository-relative paths, so it does not depend on the original Windows path from the development machine.

## Outputs

The experiment can produce:

- `results/figures/classification_map.png`
- `results/figures/confusion_matrix.png`
- `results/figures/training_history.png`
- `results/metrics/metrics.json`
- `results/metrics/y_true_1d_PU.mat`
- `results/metrics/y_pred_1d_PU.mat`
- `results/predicted_map.mat`

## Reproducibility

The experiment uses a fixed random seed (`1671`) for NumPy and Python's random module. Exact numerical reproducibility can still vary across TensorFlow versions, hardware, and execution environments.

## Important terminology note

The original notebook evaluates `S_val`/`L_val` and labels the resulting output as "Test" accuracy. In this cleaned repository these samples are consistently called **validation samples**, because the supplied experiment does not define a separate held-out test set.

## Research use

This repository is intended for research and educational use. Dataset ownership, licensing, and citation requirements should be checked before redistribution.

## Citation

If you use this implementation in academic work, cite the original Pavia University dataset according to the dataset provider's required citation and cite this repository after publication.
