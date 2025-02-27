import numpy as np
import time

from typing import Tuple

# Timing the TF import
print("\nLoading TensorFlow...")
start = time.perf_counter()

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, SimpleRNN, Flatten, LSTM, Input
from tensorflow.keras.layers import Normalization

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")


def initialize_model(input_shape: tuple, future_steps:int=12) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    model = Sequential()

    # Add an explicit Input layer
    model.add(Input(shape=input_shape))
    model.add(SimpleRNN(units=10, activation='tanh', return_sequences=False))
    model.add(Dense(20, activation="linear"))
    model.add(Dense(10, activation="linear"))
    model.add(Dense(future_steps, activation="linear"))

    print("✅ Model initialized")

    return model


def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])

    print("✅ Model compiled")

    return model

def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=256,
        patience=2,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print("\nTraining model...")

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    if validation_data:
        history = model.fit(
            X,
            y,
            validation_data=validation_data,
            epochs=100,
            batch_size=batch_size,
            callbacks=[es],
            verbose=1
        )
    else:
        history = model.fit(
            X,
            y,
            validation_split=validation_split,
            epochs=100,
            batch_size=batch_size,
            callbacks=[es],
            verbose=1
        )

    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history


def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(f"\nEvaluating model on {len(X)} rows...")

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    mae = metrics["mae"]

    print(f"✅ Model evaluated, MAE: {round(mae, 2)}")

    return metrics
