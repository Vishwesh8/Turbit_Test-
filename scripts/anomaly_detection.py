from scripts.utilities import *

"""Loading saved model"""
new_model = tf.keras.models.load_model(
    "models\LSTM_model_3_layers_multi_features.keras"
)

"""extracting and transforming data into usable format"""
t1_df = data_load_and_clean("Turbine1.csv")
t2_df = data_load_and_clean("Turbine2.csv")

"""Dimensionality reduction using PCA"""
t2_df_rel = dimensionality_reduction(
    t2_df,
    n_components=2,
    column_names=["Wind(m/s)", "Rotor(rpm)", "Strom-(A)", "Strom-.1(A)", "Strom-.2(A)"],
)

"""Preparing windows in the data and shuffling windows"""
window_size = 36
prediction_steps = 1
shift = 0

t2_inputs, t2_labels = prepare_data(
    t2_df_rel,
    ["pca1", "pca2", "Gen1-(°C)", "Lager(°C)", "Außen(°C)", "GetrT(°C)"],
    "Leistung(kW)",
    window_size,
    prediction_steps,
    shift,
    shuffle=False,
)

"""Standardising data"""
t2_inputs = (t2_inputs - t2_inputs.mean()) / t2_inputs.std()
t2_labels = (t2_labels - t2_labels.mean()) / t2_labels.std()

"""Getting predicted and actual values"""
predicted = new_model.predict(t2_inputs, batch_size=4096)
t2_labels = t2_labels.reshape(t2_labels.shape[0], t2_labels.shape[1])

"""Windowing the predicted and actual data for anomaly detection"""
window_size_ad = 18
windowed_prediction = window(predicted, window_size_ad)
windowed_actual = window(t2_labels, window_size_ad)

"""Finding prediction errors and standardising"""
std_devs = no_of_std_devs(windowed_prediction, windowed_actual)

"""To get the anomalous time intervals"""
std_devs_extended = np.repeat(std_devs, 18, axis=0)
plot_anomalies(std_devs_extended)
t2_anomaly = t2_df["Dat/Zeit()"].copy().to_frame()
t2_anomaly["std_dev"] = 0
t2_anomaly["std_dev"] = t2_anomaly["std_dev"].astype("float64")
t2_anomaly["std_dev"].iloc[-len(std_devs_extended) :] = std_devs_extended.squeeze()
t2_anomaly = t2_anomaly.loc[(t2_anomaly["std_dev"] > 3)]
t2_anomaly = t2_anomaly.drop_duplicates(subset="std_dev")
print(
    f"Start time of all the anomalous windows of three hours can be seen in this dataframe\n{t2_anomaly}"
)
