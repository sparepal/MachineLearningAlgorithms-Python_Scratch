# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:40:54 2017

@author: vikhyat
"""
#linear regression on diabetes dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from numpy import linalg as LA
from sklearn import datasets, linear_model
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.decomposition import PCA as sklPCA
from sklearn.model_selection import train_test_split

datasetDiabetes = datasets.load_diabetes()
x= datasetDiabetes.data[:, 2]
y= datasetDiabetes['target']
#Stacking&shuffling to select 20 samples for testing & rest for training
xy=np.column_stack((x,y))
xy=pd.DataFrame(xy)
xy.sample(frac=1)
# Splitting the data into training/testing sets
diabetes_train = xy[:-20]
diabetes_test = xy[-20:]
diabetes_train_x = diabetes_train[0].values.reshape(-1,1)
diabetes_test_x = diabetes_test[0].values.reshape(-1,1)
regr = linear_model.LinearRegression()
regr.fit(diabetes_train_x, diabetes_train[1])
# Make predictions using the testing set
diabetes_y_prediction = regr.predict(diabetes_test_x)
# Plotting outputs
plt.scatter(diabetes_test_x, diabetes_test[1],  color='black')
plt.plot(diabetes_test_x, diabetes_y_prediction, color='blue', linewidth=3)
plt.show()
#-----------------------------------------------------------------------------------#