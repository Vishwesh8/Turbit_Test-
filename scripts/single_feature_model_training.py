from scripts.utilities import *

"""extracting and transforming data into usable format"""
current_directory = os.getcwd()
root_directory = os.path.dirname(current_directory)
t1_df = data_load_and_clean(os.path.join(root_directory, "Turbine1.csv"))

"""Preparing windows in the data and shuffling windows before splitting the data"""
window_size = 36
prediction_steps = 1
shift = 0

t1_inputs, t1_labels = prepare_data(
    t1_df, ["Wind(m/s)"], "Leistung(kW)", window_size, prediction_steps, shift
)

""" Splitting data into train test and validation - 70% training, 20% validation, 10% testing"""
t1_train_inputs, t1_val_inputs, t1_test_inputs = data_split(t1_inputs)
t1_train_labels, t1_val_labels, t1_test_labels = data_split(t1_labels)

"""Standardising data"""
t1_train_inputs, t1_val_inputs, t1_test_inputs = data_standardization(
    t1_train_inputs, t1_val_inputs, t1_test_inputs
)
t1_train_labels, t1_val_labels, t1_test_labels = data_standardization(
    t1_train_labels, t1_val_labels, t1_test_labels
)

"""Building an LSTM based model"""
model = modelling(
    window_size,
    prediction_steps,
    t1_train_inputs.shape[-1],
    optimizer="adam",
    loss="mean_absolute_percentage_error",
)

"""Starting a training loop and saving model"""
batch_size = 4096
epochs = 1000

history, trained_model = model_training(
    model,
    t1_train_inputs,
    t1_train_labels,
    t1_val_inputs,
    t1_val_labels,
    batch_size,
    epochs,
    es_patience=100,
)

trained_model.save(
    os.path.join(root_directory, "models", "LSTM_model_2_layers_single_feature.keras")
)

"""evaluating model performance"""

loss = trained_model.evaluate(x=t1_test_inputs, y=t1_test_labels, return_dict=True)
print(f"evaluation loss = {loss}")
# predicted = trained_model.predict(t1_test_inputs, batch_size=batch_size)
