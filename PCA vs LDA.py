# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:15:42 2017

@author: vikhyat
"""
#PCA vs LDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Question 1
data=np.genfromtxt("C:/Users/vikhy/Desktop/dataset_1.csv",dtype='float',skip_header=1,delimiter=',')
v=np.column_stack((data[:,0],data[:,1]))
overall_mean = np.mean(v, axis=0)
dataC1=v[:30,:]
dataC0=v[30:61:]
#To check if there is clear seperation-found class cant be judged without the color
plt.scatter(dataC1[:,0],dataC1[:,1],color='red')
plt.scatter(dataC0[:,0],dataC0[:,1],color='blue')
#Getting PCA data and finding eigen pairs
pca_data=np.asmatrix(v)
mean_vec = np.mean(pca_data, axis=0)
cov_mat = (pca_data - mean_vec).T.dot((pca_data - mean_vec)) / (pca_data.shape[0]-1)
eig_vals,eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort()
eig_pairs.reverse()
#Projection onto PC1-Observed to be not classifying the data at all not considering the colors
pcProjection=np.dot(v,eig_vecs[0].reshape(2,1))
plt.scatter([pcProjection[0:30].reshape(30,1)],[np.zeros(30).reshape(30,1)], color="blue", alpha=0.4,marker='o', label='1')
plt.scatter([pcProjection[30:61]],[np.zeros(30)],color="red", alpha=0.4,marker='o', label='0')
plt.title('PC1 projection results')
plt.legend()
plt.show()
#PC plot
plt.plot([0,250*eig_vecs[0,1]],[0,250*eig_vecs[0,0]], color='black')
plt.plot([0,-250*eig_vecs[0,1]],[0,-250*eig_vecs[0,0]], color='black')
plt.xlim(0,35), plt.ylim(0,40)
#Doing LDA for the same
mDataC1=[float(sum(l))/len(l) for l in zip(*dataC1)]#First class(Class 1)
mDataC0=[float(sum(l))/len(l) for l in zip(*dataC0)]#Second class(Class 0)
#Calculating mean
mean_vectors = []
mean_vectors.append(mDataC1)
mean_vectors.append(mDataC0)
mean_vectors[0]
#Calculating scatter within class scatter matrix
scatter_within = np.zeros((4,4))
mean_vector_C1 = np.asarray(mean_vectors[0]).reshape(1,2)
mean_vector_C1 = np.tile(mean_vector_C1,(30,1))
mean_vector_C0 = np.asarray(mean_vectors[1]).reshape(1,2)
mean_vector_C0 = np.tile(mean_vector_C0,(30,1))
#For both classes
S_W_1 = np.matmul(np.transpose(dataC1-mean_vector_C1),(dataC1-mean_vector_C1))
S_W_2 = np.matmul(np.transpose(dataC0-mean_vector_C0),(dataC0-mean_vector_C0))
#Adding both the class's scatter matrix
S_W = np.add(S_W_1,S_W_2)
#Calculating within class scatter matrix
S_B = np.zeros((4,4))
#calculating scatter between class scatter matrix
# length of class 1 data * (np.outer(class_mean - overall_mean),(class_mean-overall_mean))
S_B_1 = 30*(np.outer((mDataC1-overall_mean),(mDataC1-overall_mean)))
S_B_2 = 30*(np.outer((mDataC0-overall_mean),(mDataC0-overall_mean)))
S_between=np.add(S_B_1,S_B_2)
#Sorting eigen values in reverse
eigen_valuesL, eigen_vectorsL = np.linalg.eig(np.dot(np.linalg.inv(S_W),S_between))
eig_pairsL = [(np.abs(eigen_valuesL[i].real), eigen_vectorsL[:,i].real) for i in range(len(eigen_valuesL))]
eig_pairsL.reverse()
W=eig_pairsL[0][1].reshape(2,1)
W = W.real
plt.plot(200*W)
ldaprojection = np.dot(v,W)
plt.scatter(ldaprojection[0:30,:],np.zeros(len(dataC1)), color="blue", alpha=0.4,marker='o', label='1')
plt.scatter(ldaprojection[30:61,:],np.zeros(len(dataC0)),color="red", alpha=0.4,marker='o', label='0')
plt.title('Results')
plt.legend()
plt.show()
#Computing variance
np.var(np.dot(v,eig_pairs[0][1]))
np.var(np.dot(v,eig_pairs[1][1]))
#Observed that higher the eigenValue higher is the variance of projection onto that eigen vector
#computing variance of projection onto W axis
np.var(np.dot(v,W))
#LDA PCA are linear transformations algorithms. PCA yields the directions (principal components) that maximize the variance of the data, whereas LDA also aims to find the directions that maximize the separation (or discrimination) between different classes, which can be useful in pattern classification problem (PCA ignores class labels). 
# PCA projects the entire dataset onto a different feature space, and LDA tries to determine a suitable feature (sub)space in order to distinguish between patterns that belong to different classes.
#That's is why we have most variance on PC1(Around 158 and very less on PC2 and LDA axis(around 5)

#Final plot
plt.scatter(dataC1[:,0],dataC1[:,1],color='g')
plt.scatter(dataC0[:,0],dataC0[:,1],color='g')
plt.plot([0,250*eig_vecs[0,1]],[0,250*eig_vecs[0,0]], color='black')
plt.plot([0,-250*eig_vecs[0,1]],[0,-250*eig_vecs[0,0]], color='black')
plt.plot(200*W)
plt.xlim(0,40)
plt.ylim(0,40)








