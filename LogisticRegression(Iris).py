# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 23:08:23 2017
@author: vikhyat
"""
#Logistic regression for classification on Iris dataset
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from __future__ import division

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def cost(zt,target):
    if target:
        return -np.log(zt)
    else:
        return  -np.log(1-zt) 
    
def split(dataset,i):
    train_data=[]
    for j in range(100):
        if j!=i:
            train_data.append(dataset[j])
    return dataset[i],train_data
    
# Load dataset
print("Loading dataset")
theta=np.zeros(2).reshape(2,1)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['petal-length', 'petal-width', 'class']
dataset = np.asmatrix(pd.read_csv(url, names=names))
dataset=dataset[50:150,:]
#Required scaling
print("Scaling dataset")
for j in range(2):
   col_min=np.min(dataset[:,j])
   col_max=np.max(dataset[:,j])
   for i in range(100):  
       dataset[i,j]=(dataset[i,j]-col_min)/(col_max-col_min)
dataset.tolist()
total_cost=0 
alpha=0.1
counter=0
test_dataset=[]
print("Training dataset")
for i in range(100):
    total_cost=0
    if i==counter:                                  #Leave-one out cross validation
        test_d,train_data=split(dataset,i)
        test_dataset.append(test_d)
        counter+=1
#Calculating H_theta(x)
    for j in range(99):                             #Has only 99 flowers leaving one for testing
        
        z=sigmoid(np.sum(np.dot(train_data[j][:,0:2],theta)))
        #Calculating respective class cost and updating theta simultaneously
        if (train_data[j][:,2]=='Iris-versicolor'):         #1 for Versicolor
            costIteration=cost(z,1)
            theta=np.subtract(theta,(alpha*(z-1)*train_data[j][:,0:2]).reshape(2,1))
        else:                                                #0 for Virginica
            costIteration=cost(z,0)
            theta=np.subtract(theta,(alpha*(z-0)*train_data[j][:,0:2]).reshape(2,1))        
        total_cost+=costIteration      
#Testing the test_dataset
print("Testing data set")
error=0
for i in range(100):
    
    z_test=1/(1+np.exp(-np.dot(test_dataset[i][:,0:2],theta)[0,0]))
    
   
    if (z_test-0)>(1-z_test):          #0 class prediction according to our trained model closer to 0/1
        if(test_dataset[i][:,2:3]=='Iris-virginica'):
            error+=1
  
average_error=error/100 
print("Average classification analysis error is")
print(average_error)
        
    

    





    
