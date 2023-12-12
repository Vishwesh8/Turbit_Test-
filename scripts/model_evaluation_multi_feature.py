from scripts.utilities import *

"""Loading saved model"""
new_model = tf.keras.models.load_model(
    "models\LSTM_model_2_layers_multi_features.keras"
)

"""extracting and transforming data into usable format"""
t1_df = data_load_and_clean("Turbine1.csv")
t2_df = data_load_and_clean("Turbine2.csv")

"""Dimensionality reduction using PCA"""
t2_df = dimensionality_reduction(
    t2_df,
    n_components=2,
    column_names=["Wind(m/s)", "Rotor(rpm)", "Strom-(A)", "Strom-.1(A)", "Strom-.2(A)"],
)

"""Preparing windows in the data and shuffling windows"""
window_size = 36
prediction_steps = 1
shift = 0

t2_inputs, t2_labels = prepare_data(
    t2_df,
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

"""Evaluating model performance"""
new_model.evaluate(x=t2_inputs, y=t2_labels, return_dict=True)

"""To visualize predictions"""

"""
plot_acf(t1_df["Leistung(kW)"], lags=500, alpha=0.05)

predicted = new_model.predict(t2_inputs, batch_size=4096)

sns.lineplot(
    x=[i for i in range(predicted.shape[0])],
    y=predicted.squeeze(),
    color="b",
    label="predicted",
)

sns.lineplot(
    x=[i for i in range(t2_labels.shape[0])],
    y=t2_labels.squeeze(),
    color="r",
    label="actual",
)
plt.legend()

# Set plot labels and title
plt.xlabel("time_steps")
plt.ylabel("value")
plt.title("Comparison")
plt.show()
"""