from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODEL_DIR = ROOT / "models"

PAVIA_IMAGE = DATA_DIR / "PaviaU.mat"
PAVIA_GT = DATA_DIR / "PaviaU_gt.mat"

SEED = 1671
TRAIN_FRACTION = 0.20
EPOCHS = 50
BATCH_SIZE = 16
DROPOUT = 0.30
LEARNING_RATE = 0.01
