import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def data_load_and_clean(filename):
    # Read data from CSV
    df = pd.read_csv(filename)

    # Add units to the column names and remove from the dataframe
    df.columns = df.columns + "(" + df.iloc[0] + ")"

    # Remove empty spaces from the column names and the units
    column_names = list(df)
    for name in column_names:
        df.rename(columns={name: name.replace(" ", "")}, inplace=True)

    # Create separate dataframes for time and for variables
    date_time_df = pd.to_datetime(
        df.iloc[1:].pop(list(df)[0]), format="%d.%m.%Y, %H:%M"
    )

    data_df = df.iloc[1:, 1:]

    # Replace all the ',' by '.' and convert the data into float format
    data_df = data_df.apply(lambda x: x.str.replace(",", "."))
    data_df = data_df.astype("float")

    data = pd.concat([date_time_df, data_df], axis=1)
    data = data.reset_index(drop=True)
    return data


def data_split(data):
    n = len(data)
    train_data = data[0 : int(n * 0.7)]
    val_data = data[int(n * 0.7) : int(n * 0.9)]
    test_data = data[int(n * 0.9) :]
    return train_data, val_data, test_data


def data_standardization(train_data, val_data, test_data):
    train_mean = train_data.mean()
    train_std = train_data.std()

    train_data = (train_data - train_mean) / train_std
    val_data = (val_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std
    return train_data, val_data, test_data


def prepare_data(
    df,
    input_column_names,
    label_column_name,
    time_window_size,
    prediction_steps,
    shift,
    shuffle=True,
):
    input_features = df[input_column_names].values
    target_feature = df[label_column_name].values

    input_data = np.zeros(
        (
            len(input_features) - time_window_size - shift - prediction_steps + 1,
            time_window_size,
            df[input_column_names].shape[-1],
        )
    )
    labels = np.zeros(
        (
            len(target_feature) - time_window_size - shift - prediction_steps + 1,
            prediction_steps,
            1,
        )
    )

    for i in range(len(df) - time_window_size - shift - prediction_steps + 1):
        window = input_features[i : i + time_window_size]
        target = target_feature[
            i
            + time_window_size
            + shift : i
            + time_window_size
            + prediction_steps
            + shift
        ]
        input_data[i] = window.reshape((time_window_size, len(input_column_names)))
        labels[i] = target.reshape((prediction_steps, 1))

    if shuffle:
        indices = np.arange(len(input_data))
        np.random.shuffle(indices)
        input_data = input_data[indices]
        labels = labels[indices]
    return input_data, labels


def modelling(window_size, prediction_steps, number_of_features, optimizer, loss):
    inputs = tf.keras.Input(shape=(window_size, number_of_features))
    x = layers.LSTM(64, activation="relu", return_sequences=True)(inputs)
    # x = layers.LSTM(64, activation="relu")(inputs)
    # x = layers.LSTM(32, activation="relu", return_sequences=True)(x)
    # x = layers.LSTM(16, activation="relu", return_sequences=True)(x)
    x = layers.LSTM(32, activation="relu")(x)
    outputs = layers.Dense(prediction_steps)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer, loss)
    print(model.summary())
    return model


def model_training(
    model,
    train_data,
    train_labels,
    val_data,
    val_labels,
    batch_size,
    epochs,
    es_patience,
):
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=es_patience, restore_best_weights=True
    )
    history = model.fit(
        train_data,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_data, val_labels),
        shuffle=True,
        verbose=1,
        callbacks=[callback],
    )
    return history, model


def convert_datetime(df, column):
    df["time"] = df[column].dt.hour + df[column].dt.minute / 60
    df["time"] = df["time"].apply(lambda x: math.sin(x * 2 * math.pi / 24))
    df["day"] = df[column].dt.dayofyear
    df["day"] = df["day"].apply(lambda x: math.sin(x * 2 * math.pi / 366))
    return df


def dimensionality_reduction(df, n_components, column_names):
    df = (df - df.mean()) / df.std()
    pca = PCA(n_components=n_components)
    pca.fit(df[column_names])
    data_pca = pca.transform(df[column_names])
    data_pca = pd.DataFrame(data_pca, columns=["pca1", "pca2"])
    df = pd.concat([df, data_pca], axis=1)
    return df


def window(array, window_size):
    num_windows = (array.shape[0] - window_size) // window_size + 1
    windowed_array = np.zeros((num_windows, window_size, array.shape[1]))
    for window in range(num_windows):
        start_index = window * window_size
        end_index = start_index + window_size
        windowed_array[window, :, :] = array[start_index:end_index, :]
    return windowed_array


def no_of_std_devs(predict, actual):
    errors = np.linalg.norm(predict - actual, axis=1)
    std_devs = (errors - errors.mean()) / errors.std()
    return std_devs


def plot_anomalies(array):
    label_encoder = LabelEncoder()
    colors = np.where(array > 3, "blue", "red")
    numeric_colors = label_encoder.fit_transform(colors)
    plt.scatter(np.arange(len(array)), array, c=numeric_colors, label="Anomaly (Blue)")
    plt.xlabel("timestep (10 minutes)")
    plt.ylabel("standardised error")
    plt.title("Error distribution against time")
    plt.show()
