import numpy as np
import time

from typing import Tuple

# Timing the TF import
print("\nLoading TensorFlow...")
start = time.perf_counter()

from tensorflow import keras
from keras import Model, optimizers
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber, LogCosh
from tensorflow.keras.layers import LSTM, Input, Dropout, Embedding, Concatenate, RepeatVector, Input, Reshape, Dropout, Dense, Attention, Bidirectional, BatchNormalization

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")



def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])

    print("✅ Model compiled")
    return model


def initialize_model(input_shape: tuple, future_steps: int = 12, num_categories: int = 21) -> Model:
    """
    Simplified Neural Network for Category-Level Forecasting:
    - Removes `county_id` input and embeddings
    - Keeps `category_id` embeddings
    - Uses fewer LSTM layers for efficiency
    - Retains Batch Normalization & Dropout for stability
    """

    # 🔹 Define Inputs (NO COUNTY)
    category_input = Input(shape=(1,), name="category_id")
    numeric_input = Input(shape=input_shape, name="numeric_features")

    # 🔹 Embedding Layer (ONLY Category)
    category_embedding = Embedding(input_dim=num_categories, output_dim=4)(category_input)
    category_embedding = Reshape((4,))(category_embedding)
    category_embedding = RepeatVector(input_shape[0])(category_embedding)  # Expand to match time steps

    # 🔹 Merge Inputs (Only Category & Numeric Features)
    merged_inputs = Concatenate(axis=-1)([category_embedding, numeric_input])

    # 🔹 Reduced LSTM Layers (Only 2)
    rnn_layer = Bidirectional(LSTM(units=48, activation="tanh", return_sequences=True))(merged_inputs)
    rnn_layer = BatchNormalization()(rnn_layer)
    rnn_layer = Dropout(0.15)(rnn_layer)

    rnn_layer = Bidirectional(LSTM(units=24, activation="tanh"))(rnn_layer)
    rnn_layer = BatchNormalization()(rnn_layer)
    rnn_layer = Dropout(0.15)(rnn_layer)

    # 🔹 Dense Layers
    dense_layer = Dense(64, activation="relu")(rnn_layer)
    dense_layer = Dense(32, activation="relu")(dense_layer)
    output_layer = Dense(future_steps, activation="linear")(dense_layer)

    # 🔹 Create Model (Only Category & Numeric Features)
    model = Model(inputs=[category_input, numeric_input], outputs=output_layer)

    print("✅ Model initialized for Category-Level Forecasting (No County)")

    return model




def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=256,
        patience=2,
        validation_data=None,
        validation_split=0.2
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
            epochs=50,
            batch_size=batch_size,
            callbacks=[es],
            verbose=1
        )
    else:
        history = model.fit(
            X,
            y,
            validation_split=validation_split,
            epochs=50,
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
