
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 4 10:20:34 2020
@author: aerler
In this program DNNRegressor is used to perform
multiple regression to predict inflation rate.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import os

months={
        "1":'January',
        "2":'Febuary',
        "3":'March',
        "4":'April',
        "5":'May',
        "6":'June',
        "7":'July',
        "8":'August',
        "9":'September',
        "10":'October',
        "11":'November',
        "12":'December'
        }

file='Monthly_Accuracy.txt'
if os.path.exists(file):
    os.remove(file)
else:
   print("Can't delete the file as it doesn't exist")

year_avg=0
for i in range(1,13):  
    # Read the CSV File
    dataset = pd.read_csv('NonAdjustedMonthlyDataset_'+str(i)+'.csv')
    y_val=dataset["cpi_core_urban_consumers"]   # Inflation dataset
    x_data=dataset.drop(['cpi_core_urban_consumers',"cpi_core_urban_consumers_exclude_food_energy","date"],axis=1)
    
    #splitting the dataset to train and test dataset
    X_train, X_eval,y_train,y_eval=train_test_split(x_data,y_val,test_size=0.3)
    
    #The MinMaxScaler function is used to scale all input value between 0 and 1.
    scaler_model = MinMaxScaler() 
    scaler_model.fit(X_train)
    X_train=pd.DataFrame(scaler_model.transform(X_train),columns=X_train.columns,index=X_train.index)
    scaler_model.fit(X_eval)
    X_eval=pd.DataFrame(scaler_model.transform(X_eval),columns=X_eval.columns,index=X_eval.index)
    
    #Creating Feature Columns
    feat_cols=[]
    dataset=dataset.drop(['cpi_core_urban_consumers',"cpi_core_urban_consumers_exclude_food_energy","date"],axis=1)
    for cols in dataset.columns:
      column=tf.feature_column.numeric_column(cols)
      feat_cols.append(column)
    
    #Deep neural network model.
    model=tf.estimator.DNNRegressor(hidden_units=[150,150,150,250],feature_columns=feat_cols,
    optimizer=lambda: tf.train.AdamOptimizer(
    learning_rate=tf.train.exponential_decay(
    learning_rate=0.001,
    global_step=tf.train.get_global_step(),
    decay_steps=10000,
    decay_rate=0.96)))
    
    #the input function
    input_func=tf.estimator.inputs.pandas_input_fn(X_train,y_train,batch_size=50,num_epochs=300,shuffle=True)
    
    model.train(input_fn=input_func,steps=300)
    train_metrics=model.evaluate(input_fn=input_func,steps=100)
    
    pred_input_func=tf.estimator.inputs.pandas_input_fn(x=X_eval,y=y_eval,batch_size=100,num_epochs=1,shuffle=False)
    preds=model.predict(input_fn=pred_input_func)
    
    predictions=list(preds)
    final_pred=[]
    for pred in predictions:
      final_pred.append(pred["predictions"])
    
    test_metric=model.evaluate(input_fn=pred_input_func,steps=1000)    
    from sklearn.metrics import r2_score
    accuracy=r2_score(y_eval,final_pred)
    year_avg =year_avg+accuracy
    ev_list=[]
    for eva1 in y_eval:
       ev_list.append(eva1)
    
    plt.scatter(y_eval,final_pred)
    f = open(file, "a")
    f.write("Accuracy for month "+months[str(i)]+":"+str(accuracy)+"\n")

f.write("\nAccuracy for Year:"+str(year_avg/12)+"\n")
f.close()