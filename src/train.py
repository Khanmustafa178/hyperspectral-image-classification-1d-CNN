import os
import random
import numpy as np
from scipy.io import savemat
import tensorflow as tf

from .config import (
    PAVIA_IMAGE, PAVIA_GT, RESULTS_DIR, FIGURES_DIR, METRICS_DIR,
    MODEL_DIR, SEED, TRAIN_FRACTION, EPOCHS, BATCH_SIZE, DROPOUT,
    LEARNING_RATE,
)
from .data import load_pavia, make_split
from .model import build_model
from .evaluate import evaluate_predictions, save_training_history, save_classification_map


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    if not PAVIA_IMAGE.exists() or not PAVIA_GT.exists():
        raise FileNotFoundError(
            "PaviaU.mat and PaviaU_gt.mat must be placed in the data/ directory. "
            "See data/README.md."
        )

    image, ground_truth = load_pavia(PAVIA_IMAGE, PAVIA_GT)
    print(f"Image shape: {image.shape}")
    print(f"Ground-truth shape: {ground_truth.shape}")

    x_train, y_train, x_val, y_val, x_all, n_classes, scene_shape = make_split(
        image, ground_truth, TRAIN_FRACTION, SEED
    )

    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Number of classes: {n_classes}")

    model = build_model(x_train.shape[1], n_classes, DROPOUT, LEARNING_RATE)
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(x_val, y_val),
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "paviaU_1d_cnn.keras")
    save_training_history(history, FIGURES_DIR)

    probabilities = model.predict(x_all, batch_size=1024, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1) + 1
    pred_map = predicted_labels.reshape(scene_shape)

    savemat(RESULTS_DIR / "predicted_map.mat", {"outData": pred_map})
    save_classification_map(pred_map, FIGURES_DIR)

    metrics = evaluate_predictions(ground_truth, pred_map, METRICS_DIR, FIGURES_DIR)
    print(f"Overall Accuracy: {metrics['overall_accuracy'] * 100:.2f}%")
    print(f"Cohen's Kappa: {metrics['kappa']:.4f}")


if __name__ == "__main__":
    main()
