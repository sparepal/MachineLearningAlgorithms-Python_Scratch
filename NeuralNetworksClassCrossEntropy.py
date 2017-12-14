# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:58:25 2017
@author: vikhyat
"""

#Neural networks for classification with cross entropy cost function on Iris data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def sigmoid(h):
    return 1 / (1 + np.exp(-h))

def NN(x,theta):
    z=np.dot(x,theta.T)
    return (z)

def cost(hTx,y):
    return -y*np.log(hTx)-(1-y)*np.log(1-hTx)           #Cross entropy cost function(derivative)
  
def split(dataset,i):
    train_data=[]
    for j in range(100):
        if j!=i:
            train_data.append(dataset[j])
    return dataset[i],train_data
print("Loading dataset")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['petal-length', 'petal-width', 'class']
dataset = np.asmatrix(pd.read_csv(url, names=names))
dataset=dataset[50:150,:]
print("Scaling dataset")
#Required scaling
for j in range(2):
   col_min=np.min(dataset[:,j])
   col_max=np.max(dataset[:,j])
   for i in range(100):  
       dataset[i,j]=(dataset[i,j]-col_min)/(col_max-col_min)
#Random initialization      
theta_1=np.asmatrix(np.random.rand(6)).reshape(2,3)
theta_2=np.asmatrix(np.random.rand(3)).reshape(1,3)
bias_column=np.asmatrix(np.ones(100)).reshape(100,1)
dataset=np.column_stack((bias_column,dataset[:,]))
indices=np.arange(100)
test_dataset=[]
y_target=[]
y_target=np.ones(100)

for i in range(100):
    if i>=49:
        y_target[i]=0

alpha=0.1
counter=0
print("Training dataset")
for i in range(100):
    
    #Leave one out cross validation
    test_d,train_data=split(dataset,i) 
    test_dataset.append(test_d)                 
    a1=dataset[:,0:3]
    #Calculating Hidden layer inputs 
    h=NN(a1,theta_1)
    a2=[]
    for i in range(100):
        a2.append((sigmoid(h[i,0]),sigmoid(h[i,1])))
    #Feeding hidden layer inputs                
    a2=np.column_stack((bias_column,np.asmatrix(a2)))
    h=NN(np.asmatrix(a2),theta_2)
    a3=[]
    for i in range(100):
        a3.append((sigmoid(h[i,0]))) 
                                  
    #Calculating outer layer gradient
    d3=a3-y_target
    delta3=np.dot(d3.reshape(1,100),a2)
    #Calculating hidden layer gradient
    a2=np.asmatrix(a2)
    d2=np.multiply(np.dot(d3.reshape(100,1),theta_2),np.multiply(a2,(1-a2))) 
    delta2=np.dot(d2.T,a1)
    delta2=delta2[1:3,:]
                
    #Updating theta(weights)
    theta_2-=alpha*delta3
    theta_1=theta_1-alpha*delta2
    
    if dataset[i,3]=='Iris-versicolor':
        total_cost=np.sum(cost(a3,1))
    else:
        total_cost=np.sum(-np.log(1-np.asmatrix(a3)) )
  

  
#Testing the data
print("Testing data set")
error=0
for i in range(100):
    a1=test_dataset[i][:,0:3]
    hT=NN(a1,theta_1)
    a2T=[]
    a2T.append((sigmoid(hT[0,0]),sigmoid(hT[0,1])))
    a2T=np.column_stack(([1],np.asmatrix(a2T)))
    hT=NN(np.asmatrix(a2T),theta_2)
    a3T=sigmoid(hT)
    
    if (a3T<0.5):   
        #0 class prediction according to our trained model
        if(test_dataset[i][:,3:4]=='Iris-virginica'):
            error+=0
            
        else:
            error+=1
            
    elif(a3T>0.5):
        
        if(test_dataset[i][:,3:4]=='Iris-versicolor'):
            error+=0
        else:
            error+=1
  
average_error=error/100 
print("Average classification analysis error is")
print(average_error)
