# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 19:00:34 2019

@author: aerler
"""

# Read the CSV File


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# library for other normilzation technique
#from sklearn.preprocessing import StandardScaler
import seaborn as sns
#Calling of data 
dataset = pd.read_csv('Final_Dataset_3.csv')
data = dataset.drop(['CPI','CPI_Core','year','Unnamed: 0'],axis=1)
#setup for PCA transformation
pca_trafo = PCA().fit(data);
#Normilzation for data
z_scaler = MinMaxScaler()

#Example plot of PCA without normilzation
#Without Normilzing data we have a large issue with bias
plt.figure(figsize = (10,6.5));
plt.semilogy(pca_trafo.explained_variance_ratio_, '--o');
plt.xlabel('principal component', fontsize = 20);
plt.ylabel('explained variance', fontsize = 20);
plt.tick_params(axis='both', which='major', labelsize=18);
plt.tick_params(axis='both', which='minor', labelsize=12);
plt.xlim([0, 16]);



plt.figure(figsize = (10,6.5));
plt.semilogy(np.square(data.std(axis=0)) / np.square(data.std(axis=0)).sum(), '--o', label = 'variance ratio');
plt.semilogy(data.mean(axis=0) / np.square(data.mean(axis=0)).sum(), '--o', label = 'mean ratio');
plt.xlabel('original feature', fontsize = 20);
plt.ylabel('variance', fontsize = 20);
plt.tick_params(axis='both', which='major', labelsize=18);
plt.tick_params(axis='both', which='minor', labelsize=12);
plt.xlim([0, 16]);
plt.legend(loc='lower left', fontsize=18);



#Normalize Data
z_data = z_scaler.fit_transform(data)
#Apply PCA to normalized data
pca_trafo = PCA().fit(z_data);
#Creat a plot to show exaplined and cumlative variance per PCA component
fig, ax1 = plt.subplots(figsize = (10,6.5))
ax1.semilogy(pca_trafo.explained_variance_ratio_, '--o', label = 'explained variance ratio');
color =  ax1.lines[0].get_color()
ax1.set_xlabel('principal component', fontsize = 20);

    
plt.legend(loc=(0.01, 0.075) ,fontsize = 18);

ax2 = ax1.twinx()
ax2.semilogy(pca_trafo.explained_variance_ratio_.cumsum(), '--go', label = 'cumulative explained variance ratio');
for tl in ax2.get_yticklabels():
    tl.set_color('g')

ax1.tick_params(axis='both', which='major', labelsize=18);
ax1.tick_params(axis='both', which='minor', labelsize=12);
ax2.tick_params(axis='both', which='major', labelsize=18);
ax2.tick_params(axis='both', which='minor', labelsize=12);
plt.xlim([0, 16]);
plt.legend(loc=(0.01, 0),fontsize = 18);


n_comp =14 #Number of attributes
pca_trafo = PCA(n_components=n_comp)#transforms pca for n_comp

# Standardize or Normalize every column in the figure
# Standardize:
#sns.clustermap(df, standard_scale=1)
# Normalize

sns.clustermap(data, z_score=1) #Clusertmap

pca_data = pca_trafo.fit_transform(data)
pca_inv_data = pca_trafo.inverse_transform(np.eye(n_comp))

#Create heatmap for componenets 
fig = plt.figure(figsize=(10, 6.5))
sns.heatmap(np.log(pca_trafo.inverse_transform(np.eye(n_comp))), cmap="hot")
plt.ylabel('principal component', fontsize=20);
plt.xlabel('original feature index', fontsize=20);
plt.tick_params(axis='both', which='major', labelsize=18);
plt.tick_params(axis='both', which='minor', labelsize=12);

#Plot Mean and variance of data
fig = plt.figure(figsize=(10, 6.5))
#plt.plot(pca_inv_data.mean(axis=0), '--o', label = 'mean')
plt.plot(np.square(pca_inv_data.std(axis=0)), '--o', label = 'variance')
plt.legend(loc='lower right')
plt.ylabel('feature contribution', fontsize=20);
plt.xlabel('feature index', fontsize=20);
plt.tick_params(axis='both', which='major', labelsize=18);
plt.tick_params(axis='both', which='minor', labelsize=12);
plt.xlim([0, 16])
plt.legend(loc='lower left', fontsize=18)

data_scaled = pd.DataFrame(z_data,columns = data.columns) 

# PCA
pca = PCA(n_components=14)
pca.fit_transform(data_scaled)

# Dump components relations with features:
test = pd.DataFrame(pca.components_,columns=data_scaled.columns,index = 
                    ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6','PC-7','PC-8','PC-9','PC-10','PC-11','PC-12','PC-13','PC-14'])
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print (test)

fig = plt.figure(figsize=(10, 6.5))
sns.heatmap(test, cmap="hot")
plt.ylabel('principal component', fontsize=20);
plt.xlabel('original feature index', fontsize=20);
plt.tick_params(axis='both', which='major', labelsize=18);
plt.tick_params(axis='both', which='minor', labelsize=12);