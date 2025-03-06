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
from tensorflow.keras.layers import Dense, SimpleRNN, Flatten, Input, Dropout, Embedding, Concatenate

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")


def initialize_model(input_shape: tuple, future_steps:int=12) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    model = Sequential()

    # Better RNN layers with return_sequences=True
    #model.add(Input(shape=input_shape))
    model.add(SimpleRNN(units=64, activation='tanh', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(units=32, activation='tanh'))
    model.add(Dropout(0.2))
    # More complex Dense layers
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(future_steps, activation="linear"))

    print("✅ Model initialized")

    return model


def initialize_model_fail(input_shape: tuple, future_steps: int = 12, num_countries=6, num_categories=7) -> Model:
    """
    Initialize the model with Embedding layers for categorical features
    """
    county_input = Input(shape=(1,))
    category_input = Input(shape=(1,))
    time_series_input = Input(shape=input_shape)  # (past_steps, features)

    # Embedding layers for categorical variables
    county_embedding = Embedding(input_dim=num_countries, output_dim=4)(county_input)
    category_embedding = Embedding(input_dim=num_categories, output_dim=4)(category_input)

    county_embedding = Flatten()(county_embedding)
    category_embedding = Flatten()(category_embedding)

    # Process the time series data
    rnn_out = SimpleRNN(units=64, activation='tanh', return_sequences=True)(time_series_input)
    rnn_out = Dropout(0.2)(rnn_out)
    rnn_out = SimpleRNN(units=32, activation='tanh')(rnn_out)
    rnn_out = Dropout(0.2)(rnn_out)

    # Combine all features
    merged = Concatenate()([rnn_out, county_embedding, category_embedding])

    # Dense layers
    x = Dense(128, activation="relu")(merged)
    x = Dense(64, activation="relu")(x)
    output = Dense(future_steps, activation="linear")(x)

    # Define the model
    model = Model(inputs=[time_series_input, county_input, category_input], outputs=output)

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
        patience=10,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print("\nTraining model...")

    es = EarlyStopping(
        monitor="val_mae",
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
