# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 22:11:03 2017
@author: vikhyat
"""
#PCA vs Linear regression() 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from numpy import linalg as LA
from sklearn import datasets, linear_model
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.decomposition import PCA as sklPCA
from sklearn.model_selection import train_test_split
#---------------------------------------------------------------------------------#
data=np.genfromtxt("C:/Users/vikhy/Desktop/linear_regression_test_data.csv",dtype='float',skip_header=1,delimiter=',')
data=np.delete(data,0,axis=1)
pca_data=data[:,:2]
pca_data=np.asmatrix(pca_data)
mean_vec = np.mean(pca_data, axis=0)
cov_mat = (pca_data - mean_vec).T.dot((pca_data - mean_vec)) / (pca_data.shape[0]-1)
eig_vals,eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort()
eig_pairs.reverse()
#Getting columns to plot b/w x,y and x & yth
x= data[:,0] 
y= data[:,1] 
y_th= data[:,2]
#Plotting
plt.scatter(x,y,color="red")
plt.scatter(x,y_th,color="blue")
plt.plot([0,250*eig_vecs[0,1]],[0,250*eig_vecs[0,0]], color='yellow')
plt.plot([0,-250*eig_vecs[0,1]],[0,-250*eig_vecs[0,0]], color='yellow')
plt.xlim(-6,6), plt.ylim(-6,6)
plt.show()
#-----------------------------------------------------------------------------------#
x_mean=np.mean(x)
y_mean=np.mean(y)
x_variance=np.var(x)
y.shape
xy_cov_mat=np.cov(x,y)
xy_cov=xy_cov_mat[1,0]
b1=(xy_cov)/x_variance
b0=(y_mean-(b1*x_mean))
xreg=np.linspace(-100,100,100)
yreg= b0 + (b1*xreg)
#plotting with the PCA 
plt.plot(xreg,yreg,color="black")
plt.scatter(x,y,color="red")
plt.scatter(x,y_th,color="blue")
plt.plot([0,250*eig_vecs[0,1]],[0,250*eig_vecs[0,0]], color='yellow')
plt.plot([0,-250*eig_vecs[0,1]],[0,-250*eig_vecs[0,0]], color='yellow')
plt.xlim(-6,6), plt.ylim(-6,6)
plt.show()
#The PC1 axis and the regression line have been observed to be very similar
#-----------------------------------------------------------------------------------#

