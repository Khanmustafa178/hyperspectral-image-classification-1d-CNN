import random
import numpy as np
from scipy.io import loadmat
from tensorflow.keras.utils import to_categorical


def load_pavia(image_path, ground_truth_path):
    """Load PaviaU hyperspectral cube and ground-truth map."""
    image = loadmat(image_path)["paviaU"]
    ground_truth = loadmat(ground_truth_path)["paviaU_gt"]
    return image, ground_truth


def make_split(image, ground_truth, train_fraction=0.20, seed=1671):
    """Create the same class-wise random split used in the supplied notebook.

    For every labeled class, train_fraction of pixels are selected for training;
    the remaining labeled pixels are used for validation.
    """
    rng = random.Random(seed)
    rows, cols, bands = image.shape
    pixels = image.reshape(rows * cols, bands)
    labels = ground_truth.reshape(-1)
    n_classes = int(labels.max())

    train_samples, train_labels = [], []
    val_samples, val_labels = [], []

    for class_id in range(1, n_classes + 1):
        indices = np.flatnonzero(labels == class_id)
        n_train = int(len(indices) * train_fraction)
        selected_positions = rng.sample(range(len(indices)), n_train)
        selected = indices[selected_positions]
        validation = np.delete(indices, selected_positions)

        train_samples.append(pixels[selected])
        train_labels.append(np.full(n_train, class_id - 1, dtype=np.int64))
        val_samples.append(pixels[validation])
        val_labels.append(np.full(len(validation), class_id - 1, dtype=np.int64))

    x_train = np.vstack(train_samples)
    y_train = np.concatenate(train_labels)
    x_val = np.vstack(val_samples)
    y_val = np.concatenate(val_labels)

    x_train = normalize_spectra(x_train)
    x_val = normalize_spectra(x_val)
    x_all = normalize_spectra(pixels)

    x_train = x_train[..., np.newaxis]
    x_val = x_val[..., np.newaxis]
    x_all = x_all[..., np.newaxis]

    y_train = to_categorical(y_train, n_classes)
    y_val = to_categorical(y_val, n_classes)

    return x_train, y_train, x_val, y_val, x_all, n_classes, (rows, cols)


def normalize_spectra(x):
    """Match the supplied notebook's /255 spectral normalization."""
    return x.astype("float32") / 255.0
