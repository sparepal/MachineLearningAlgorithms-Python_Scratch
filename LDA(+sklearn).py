# -*- coding: utf-8 -*-
"""
Created on Sat Dec 09 12:02:44 2017
@author: vikhyat
"""
#LDA also with sklearn on SCLC_study_output_filtered_2.csv
import numpy as np
from numpy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from matplotlib import pyplot as pl
#----------------------------------------------------------------------------------#
data=np.genfromtxt("SCLC_study_output_filtered_2.csv",dtype='float',skip_header=1,delimiter=',')
data=np.delete(data,0,axis=1)
dataOG=data
#Storing overall mean for inter class scatter matrix
overall_mean = np.mean(data, axis=0)
#Seperating class data
dataNSCLC=data[0:20,0:]
dataSCLC=data[20:,0:]
data = []
data.append(dataNSCLC)
data.append(dataSCLC)
#Calculating mean 
mDataNSCLC=[float(sum(l))/len(l) for l in zip(*dataNSCLC)]
mDataSCLC=[float(sum(l))/len(l) for l in zip(*dataSCLC)]
mean_vectors = []
mean_vectors.append(mDataNSCLC)
mean_vectors.append(mDataSCLC)
#Calculating scatter within class scatter matrix
scatter_within = np.zeros((19,19))
mean_vector_NSCLC = np.asarray(mean_vectors[0]).reshape(1,19)
mean_vector_NSCLC = np.tile(mean_vector_NSCLC,(20,1))
mean_vector_SCLC = np.asarray(mean_vectors[1]).reshape(1,19)
mean_vector_SCLC = np.tile(mean_vector_SCLC,(20,1))
#For both classes
S_W_1 = np.matmul(np.transpose(data[0]-mean_vector_NSCLC),(data[0]-mean_vector_NSCLC))
S_W_2 = np.matmul(np.transpose(data[1]-mean_vector_SCLC),(data[1]-mean_vector_SCLC))
#Adding both the class's scatter matrix
S_W = np.add(S_W_1,S_W_2)
#Calculating within class scatter matrix
S_B = np.zeros((19,19))
#calculating scatter between class scatter matrix
# length of class 1 data * (np.outer(class_mean - overall_mean),(class_mean-overall_mean))
S_B_1 = 20*(np.outer((mDataNSCLC-overall_mean),(mDataNSCLC-overall_mean)))
S_B_2 = 20*(np.outer((mDataSCLC-overall_mean),(mDataSCLC-overall_mean)))
S_between=np.add(S_B_1,S_B_2)
#Sorting eigen values in reverse
eigen_values, eigen_vectors = np.linalg.eig(np.dot(np.linalg.inv(S_W),S_between))
eig_pairs = [(np.abs(eigen_values[i].real), eigen_vectors[:,i].real) for i in range(len(eigen_values))]
eig_pairs.reverse()
W=eig_pairs[0][1].reshape(19,1)
W = W.real
ldaprojection = np.dot(dataOG,W)
pl.scatter(ldaprojection[0:20,:],np.zeros(len(dataNSCLC)), color="blue", alpha=0.4,marker='o', label='NSCLC')
pl.scatter(ldaprojection[20:40,:],np.zeros(len(dataSCLC)),color="red", alpha=0.4,marker='o', label='SCLC')
pl.title('Results')
pl.legend()
pl.show()
#LDA sklearn
X= dataOG
y= np.concatenate((np.zeros((1,20)),np.ones((1,20))),axis=1).reshape(40,1)
# apply sklearn LDA to cell line data
sklearn_LDA = LDA(n_components=1)
sklearn_LDA_projection = sklearn_LDA.fit_transform(X, y)
# plot the projections
pl.scatter(sklearn_LDA_projection[0:20,:], np.zeros(len(X)/2), marker='o', color='blue', label='NSCLC')
pl.scatter(sklearn_LDA_projection[20:40,:], np.zeros(len(X)/2), marker='o',  color='red', label='SCLC')
pl.title('Results from applying sklearn LDA to cell line data')
pl.xlabel(r'$W_1$')
pl.ylabel('')
pl.legend()
pl.show()