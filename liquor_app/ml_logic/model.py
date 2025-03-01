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
from tensorflow.keras.layers import SimpleRNN, Flatten, LSTM, Input, Dropout, Embedding, Concatenate, RepeatVector, Input, Reshape, LSTM, Dropout, Dense
from tensorflow.keras.layers import Normalization

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")


from tensorflow.keras.layers import Embedding, Concatenate, RepeatVector, Input, Reshape, LSTM, Dropout, Dense

def initialize_model(input_shape: tuple, future_steps: int = 12, num_counties: int = 11, num_categories: int = 21) -> Model:
    """
    Initialize the Neural Network:
    - Adds Embedding layers for county and category.
    - Expands embeddings to match time steps (52).
    """

    # Define inputs
    county_input = Input(shape=(1,), name="county_id")  # Single number input for county
    category_input = Input(shape=(1,), name="category_id")  # Single number input for category
    numeric_input = Input(shape=input_shape, name="numeric_features")  # Shape: (52, num_features)

    # Add Embeddings
    county_embedding = Embedding(input_dim=num_counties, output_dim=4)(county_input)  # Shape: (None, 1, 4)
    category_embedding = Embedding(input_dim=num_categories, output_dim=4)(category_input)  # Shape: (None, 1, 4)

    # Fix: Ensure correct reshape (remove extra dimension)
    county_embedding = Reshape((4,))(county_embedding)  # Now (None, 4)
    category_embedding = Reshape((4,))(category_embedding)  # Now (None, 4)

    # Fix: Use `RepeatVector()` with correct shape
    county_embedding = RepeatVector(input_shape[0])(county_embedding)  # Now (None, 52, 4)
    category_embedding = RepeatVector(input_shape[0])(category_embedding)  # Now (None, 52, 4)

    # Merge inputs (numeric + embeddings)
    merged_inputs = Concatenate(axis=-1)([county_embedding, category_embedding, numeric_input])  # Shape: (None, 52, num_features + 8)

    # RNN Layers
    rnn_layer = LSTM(units=64, activation="tanh", return_sequences=True)(merged_inputs)
    rnn_layer = Dropout(0.1)(rnn_layer)
    rnn_layer = LSTM(units=32, activation="tanh")(rnn_layer)
    rnn_layer = Dropout(0.1)(rnn_layer)

    # Fully Connected Layers
    dense_layer = Dense(128, activation="relu")(rnn_layer)
    dense_layer = Dense(64, activation="relu")(dense_layer)
    output_layer = Dense(future_steps, activation="linear")(dense_layer)

    # Create Model
    model = Model(inputs=[county_input, category_input, numeric_input], outputs=output_layer)

    print("✅ Model initialized with expanded embeddings")

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
