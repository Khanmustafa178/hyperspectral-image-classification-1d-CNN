from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.optimizers import SGD


def build_model(n_bands, n_classes, dropout=0.30, learning_rate=0.01):
    """Build the two-block 1D CNN used in the supplied experiment."""
    model = Sequential([
        Input(shape=(n_bands, 1)),
        Conv1D(filters=32, kernel_size=5, activation="relu"),
        Dropout(dropout),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=5, activation="relu"),
        Dropout(dropout),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=SGD(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model
