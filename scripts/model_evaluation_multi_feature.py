from scripts.utilities import *

"""Loading saved model"""
current_directory = os.getcwd()
root_directory = os.path.dirname(current_directory)
new_model = tf.keras.models.load_model(
    os.path.join(root_directory, "models", "LSTM_model_3_layers_multi_feature.keras")
)

"""extracting and transforming data into usable format"""
t1_df = data_load_and_clean(os.path.join(root_directory, "Turbine1.csv"))
t2_df = data_load_and_clean(os.path.join(root_directory, "Turbine2.csv"))

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
std_t2_inputs = (t2_inputs - t2_inputs.mean()) / t2_inputs.std()
std_t2_labels = (t2_labels - t2_labels.mean()) / t2_labels.std()

"""Evaluating model performance"""
new_model.evaluate(x=std_t2_inputs, y=std_t2_labels, return_dict=True)

"""To visualize predictions"""

std_predicted = new_model.predict(std_t2_inputs, batch_size=4096)
predicted = t2_labels.mean() + std_predicted * t2_labels.std()

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
plt.xlabel("time_steps (10 minute interval)")
plt.ylabel("Leistung(kW)")
plt.title("Comparison Between Forecasted and Actual Values (Multi-feature input)")
plt.show()
