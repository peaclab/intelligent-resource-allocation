"""
    -This script provides the first baseline [Thonglek et al.,CLUSTER'19] for the resource allocation problem we solve.
    -The baseline model is an LSTM-based model that predicts the suitable CPU and memory allocation for a given workload.
    
    -The model architecture is based on the following paper:
    "Thonglek, K., Ichikawa, K., Takahashi, K., Iida, H., & Nakasan, C. (2019, September). 
    Improving resource utilization in data centers using an LSTM-based prediction model. 
    In 2019 IEEE international conference on cluster computing (CLUSTER) (pp. 1-8). IEEE."

"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Multiply, TimeDistributed, Concatenate)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def build_lstm_model(feature_names):
    """
    Build an LSTM-based model for predicting suitable CPU and memory allocation for a given workload.

    Args:
        feature_names (list): A list of feature names to be used in the model.

    Returns:
        model: A compiled LSTM model for predicting suitable CPU and memory allocation.
    """

    input_dim = 4 # Expected number of input features is 4
    time_steps = None  # Variable-length time series
    memory_cell_size = 60  # Best-performing memory-cell size is found to be 60 in the cited paper.

    # Input layer
    inputs = Input(shape=(time_steps, input_dim), name="Input_Layer")

    # First LSTM layer: CPU and memory correlation
    lstm1_cpu = LSTM(memory_cell_size, return_sequences=True, name="LSTM1_CPU")(inputs)
    lstm1_mem = LSTM(memory_cell_size, return_sequences=True, name="LSTM1_Memory")(inputs)

    # Concatenate outputs of the first LSTM layer
    concat_lstm1 = Concatenate(name="Concatenate_LSTM1")([lstm1_cpu, lstm1_mem])

    # Multiplication layer: resource type and utilization relationship
    multiplied = Multiply(name="Multiplication_Layer")([lstm1_cpu, lstm1_mem])

    # Second LSTM layer: allocated vs used resources
    lstm2_alloc = LSTM(memory_cell_size, return_sequences=True, name="LSTM2_Allocated")(multiplied)
    lstm2_used = LSTM(memory_cell_size, return_sequences=True, name="LSTM2_Used")(multiplied)

    # Concatenate outputs of the second LSTM layer
    concat_lstm2 = Concatenate(name="Concatenate_LSTM2")([lstm2_alloc, lstm2_used])

    # Fully connected layer
    fc = TimeDistributed(Dense(memory_cell_size, activation="relu"), name="Fully_Connected_Layer")(concat_lstm2)

    # Output layer: Suitable CPU and memory allocation
    cpu_output = TimeDistributed(Dense(1, activation="linear"), name="CPU_Output")(fc)
    mem_output = TimeDistributed(Dense(1, activation="linear"), name="Memory_Output")(fc)

    # Combine outputs
    outputs = Concatenate(name="Output_Layer")([cpu_output, mem_output])

    # Define and compile the model
    model = Model(inputs=inputs, outputs=outputs, name="LSTM_Prediction_Model")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), 
                  loss="mean_squared_error", 
                  metrics=["mean_absolute_error"])

    return model


def train_lstm_model(dataframe, feature_names, target_columns, time_steps=10, test_size=0.2, batch_size=32, epochs=10):
    """
    This function prepares the data, builds the LSTM model, and performs training.

    Args:
        dataframe (DataFrame): Input dataframe containing the time-series data.
        feature_names (list): List of column names to be used as features.
        target_columns (list): List of column names to be used as targets.
        test_size (float, optional): Test to train ratio. Defaults to 0.2.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        epochs (int, optional): Number of training epochs. Defaults to 10.

    Returns:
        model (tf.keras.Model): Trained LSTM prediction model.
        history (History): Training history object.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test targets.
    """
    dataframe = dataframe.dropna()

    # Extract features and targets
    features = dataframe[feature_names].values
    targets = dataframe[target_columns].values

    # Normalize data
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    features = scaler_x.fit_transform(features)
    targets = scaler_y.fit_transform(targets)

    # Reshape data for LSTM
    num_samples = features.shape[0] - time_steps + 1
    features = np.array([features[i:i + time_steps] for i in range(num_samples)])
    targets = np.array([targets[i:i + time_steps] for i in range(num_samples)])

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=test_size, random_state=42)

    # Build the model
    model = build_lstm_model(feature_names)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs
    )

    return model, history, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    This function evaluates the trained model on the test set.

    Args:
        model (tf.keras.Model): Trained LSTM model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): True test targets.

    Returns:
        rmse (int): RMSE value.
        mae (int): MAE value.
        r2 (int): R-squared value.
    """

    # Predict on test set
    y_pred = model.predict(X_test)

    # Flatten predictions and true values
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    y_test_flat = y_test.reshape(-1, y_test.shape[-1])

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
    r2 = r2_score(y_test_flat, y_pred_flat)
    mae = mean_absolute_error(y_test_flat, y_pred_flat)

    return rmse, r2, mae
