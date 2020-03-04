import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import operator
import os
import shutil

# Current dataset to build model with
current_dataset = "NonAdjustedMonthlyDataset.csv"

# Function to create and train a dnnlinearcombinedregressor with variable number of neurons and hidden layers
def create_and_train_dnnlinearcombinedregressor (num_neurons, num_hidden_layers):
    # Read the CSV File
    df = pd.read_csv(current_dataset)
    
    # Split dataset into y (target) and x (training features)
    y_val = df["cpi_core_urban_consumers"]
    x_data = df.drop(["cpi_core_urban_consumers", "cpi_core_urban_consumers_exclude_food_energy", "date"], axis=1)
    
    # Split the dataset into training (70%) and test data (30%)
    X_train, X_eval, y_train, y_eval = train_test_split(x_data, y_val, test_size=0.3)
    
    # Instantiate a MinMaxScaler and use fit and transform to scale data
    scaler_model = MinMaxScaler()
    X_train_scaled = scaler_model.fit_transform(X_train)
    X_eval_scaled = scaler_model.fit_transform(X_eval)
    
    # Making X_train and X_eval dataframes
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_eval = pd.DataFrame(X_eval_scaled, columns=X_eval.columns, index=X_eval.index)
    
    # Creating feature columns
    feat_cols = []
    df = df.drop(["cpi_core_urban_consumers", "cpi_core_urban_consumers_exclude_food_energy", "date"], axis=1)
    for cols in df.columns:
        column = tf.feature_column.numeric_column(cols)
        feat_cols.append(column)

    # Setting up the hidden units
    hidden_units = []
    for i in range(num_hidden_layers):
        hidden_units.append(num_neurons)
    
    # Setting up the model
    model = tf.estimator.DNNLinearCombinedRegressor(model_dir="test_dnnlinearcombinedregressor", dnn_hidden_units=hidden_units, dnn_feature_columns=feat_cols, dnn_optimizer=lambda: tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(learning_rate=0.001, global_step=tf.train.get_global_step(), decay_steps=10000, decay_rate=0.96)))
    
    # Setting up the input function
    input_func = tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size=50, num_epochs=300, shuffle=True)
    
    # Training the model
    model.train(input_fn=input_func, steps=300)
    
    # Evaluate the model's accuracy
    train_metrics = model.evaluate(input_fn=input_func, steps=300)
    
    # Setting up the prediction input function
    pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_eval, y=y_eval, batch_size=4, num_epochs=1, shuffle=False)
    
    # Predict values
    preds = model.predict(input_fn=pred_input_func)
    
    # Store final predictions
    predictions = list(preds)
    final_pred = []
    for pred in predictions:
        final_pred.append(pred["predictions"])
        
    from sklearn.metrics import r2_score
    accuracy = r2_score(y_eval, final_pred)

    # Removing the files and directories to the model for later use with other models
    folder = 'test_dnnlinearcombinedregressor'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

    # Return the Accuracy of the model
    return accuracy

# Function to create and train a dnnregressor with variable number of neurons and hidden layers
def create_and_train_dnnregressor (num_neurons, num_hidden_layers):
    # Read the CSV File
    df = pd.read_csv(current_dataset)
    
    # Split dataset into y (target) and x (training features)
    y_val = df["cpi_core_urban_consumers"]
    x_data = df.drop(["cpi_core_urban_consumers", "cpi_core_urban_consumers_exclude_food_energy", "date"], axis=1)
    
    # Split the dataset into training (70%) and test data (30%)
    X_train, X_eval, y_train, y_eval = train_test_split(x_data, y_val, test_size=0.3)
    
    # Instantiate a MinMaxScaler and use fit and transform to scale data
    scaler_model = MinMaxScaler()
    X_train_scaled = scaler_model.fit_transform(X_train)
    X_eval_scaled = scaler_model.fit_transform(X_eval)
    
    # Making X_train and X_eval dataframes
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_eval = pd.DataFrame(X_eval_scaled, columns=X_eval.columns, index=X_eval.index)
    
    # Creating feature columns
    feat_cols = []
    df = df.drop(["cpi_core_urban_consumers", "cpi_core_urban_consumers_exclude_food_energy", "date"], axis=1)
    for cols in df.columns:
        column = tf.feature_column.numeric_column(cols)
        feat_cols.append(column)

    # Setting up the hidden units
    hidden_units_list = []
    for i in range(num_hidden_layers):
        hidden_units_list.append(num_neurons)
    
    # Setting up the model
    model = tf.estimator.DNNRegressor(model_dir="test_dnnregressor", hidden_units=hidden_units_list,feature_columns=feat_cols,
    optimizer=lambda: tf.train.AdamOptimizer(
        learning_rate=tf.train.exponential_decay(
            learning_rate=0.001, global_step=tf.train.get_global_step(), decay_steps=10000, decay_rate=0.96)))
    
    # Setting up the input function
    input_func = tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size=50, num_epochs=300, shuffle=True)
    
    # Training the model
    model.train(input_fn=input_func, steps=300)
    
    # Evaluate the model's accuracy
    train_metrics = model.evaluate(input_fn=input_func, steps=300)
    
    # Setting up the prediction input function
    pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_eval, y=y_eval, batch_size=4, num_epochs=1, shuffle=False)
    
    # Predict values
    preds = model.predict(input_fn=pred_input_func)
    
    # Store final predictions
    predictions = list(preds)
    final_pred = []
    for pred in predictions:
        final_pred.append(pred["predictions"])
        
    from sklearn.metrics import r2_score
    accuracy = r2_score(y_eval, final_pred)

    # Removing the files and directories to the model for later use with other models
    folder = 'test_dnnregressor'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

    # Return the Accuracy of the model
    return accuracy


# Creating the ranges under test (start, stop, step)
neurons_range = range(50, 300, 25)
hidden_layers_range = range(2, 5, 1)

# Specifiying how many repetitions should be performed before averaging the accuracies
num_repetitions = 10

# Creating a dictionary to store accuracies for the dnnregressor
accuracy_dict_dnnregressor = {}

# for each amount of hidden layers in the range, create and train model
for amount_hidden_layers in hidden_layers_range:
    # for each amount of neurons in the range, create and train model
    for amount_neurons in neurons_range:
        # variable to hold the average accuracy
        average_accuracy = 0

        # calculating the average accuracy with this number of hidden layers and neurons
        for i in range(num_repetitions):
            average_accuracy = average_accuracy + create_and_train_dnnregressor(amount_neurons, amount_hidden_layers)
        
        average_accuracy = average_accuracy / num_repetitions
        
        # creating dictionary entry and storing accuracy
        accuracy_dict_key = "Hidden Layers: " + str(amount_hidden_layers) + " Neurons: " + str(amount_neurons)
        accuracy_dict_dnnregressor[accuracy_dict_key] = average_accuracy


# Creating a dictionary to store accuracies for the dnnlinearcombinedregressor
accuracy_dict_linear_combined = {}

# for each amount of hidden layers in the range, create and train model
for amount_hidden_layers in hidden_layers_range:
    # for each amount of neurons in the range, create and train model
    for amount_neurons in neurons_range:
        # variable to hold the average accuracy
        average_accuracy = 0

        # calculating the average accuracy with this number of hidden layers and neurons
        for i in range(num_repetitions):
            average_accuracy = average_accuracy + create_and_train_dnnlinearcombinedregressor(amount_neurons, amount_hidden_layers)
        
        average_accuracy = average_accuracy / num_repetitions
        
        # creating dictionary entry and storing accuracy
        accuracy_dict_key = "Hidden Layers: " + str(amount_hidden_layers) + " Neurons: " + str(amount_neurons)
        accuracy_dict_linear_combined[accuracy_dict_key] = average_accuracy

# Getting the combination of hidden layers and neurons with the highest accuracy
best_accuracy_dnnregressor = max(accuracy_dict_dnnregressor.items(), key=operator.itemgetter(1))[0]
best_accuracy_linear_combined = max(accuracy_dict_linear_combined.items(), key=operator.itemgetter(1))[0]
print("The best accuracy for regressor only is with " + best_accuracy_dnnregressor + " with an accuracy of " + str(accuracy_dict_dnnregressor[best_accuracy_dnnregressor]))
print("The best accuracy for linear combined is with " + best_accuracy_linear_combined + " with an accuracy of " + str(accuracy_dict_linear_combined[best_accuracy_linear_combined]))
