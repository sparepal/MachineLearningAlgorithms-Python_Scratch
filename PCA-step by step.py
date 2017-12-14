import numpy as np
from numpy import genfromtxt
from numpy import linalg as lg
from matplotlib import pyplot as plt

#importing the dataset 
data_1 = genfromtxt("C:\Users\vikhy\Downloads\SCLC_study_output_filtered.csv", delimiter=",")
data_1

#deleting the first column with names
data_temp1 = np.delete(data_1,(0),axis=0)
#deleteing first row with headings
data= np.delete(data_temp1,0,axis=1)

#converting numpy array to numpy matrix
data = np.asmatrix(data)
data
type(data)
data.shape

#transpose of matrix data
data_transpose = data.transpose()
data_transpose.shape


#calculating covariance matrix
covariance_matrix = np.divide(((data).T.dot(data).T),len(data))
covariance_matrix
sum(np.diagonal(covariance_matrix))



#1. Total Variance of original variables
np.diagonal(covariance_matrix)
print('Covariance Matrix \n%s'%covariance_matrix)


#performing eigen value decomposition
eigen_vals, eigen_vecs = np.linalg.eig(np.cov(data.T))
print('Eigen Vectors \n %s' %eigen_vecs)
print('Eigen Values \n %s' %eigen_vals)
eigen_vals.shape

np.savetxt("eigen values.csv", eigen_vals, delimiter=",")

# create a list of tuple (eigenvalue, eigenvector) 
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

np.diagonal(eigen_vecs)

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda x: x[0], reverse=True)
eigen_pairs
np.savetxt("eigen pairs.csv", eigen_pairs, delimiter=",")


# checking the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eigen_pairs:
    eigen=i[0]
    print eigen


#explained variance    
#4. It can be noted from the cum_var_exp array that we need 4 prinicpal components to keep 75% of the variance
total = sum(eigen_vals)
var_exp = [(i / total)*100 for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
cum_var_exp

#plotting prinicpal components
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(7, 7))

    plt.bar(range(3), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
   
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    

#projection matrix:selecting first two pc
matrix_w = np.hstack((eigen_pairs[0][1].reshape(49,1),
                      eigen_pairs[1][1].reshape(49,1)))

print("projection matrix:\n%s"%matrix_w) 



# 2.) covariance of first PC and second PC
cov_pc = np.cov(matrix_w[0:49,0],matrix_w[0:49,1])
cov_pc


# projecting to new space
Y = data.dot(matrix_w)
Y.shape

#p3.) plotting scores between first and second pc  in red AND blue
plt.plot(Y[0:20,0],Y[0:20,1], 'o', markersize=7, color='red', alpha=0.5, label='class1')
plt.plot(Y[20:40,0],Y[20:40,1], '^', markersize=7, color='blue', alpha=0.5, label='class2')
plt.xlim([-100000000000000000000000000,1000000000000000000000000000])
plt.ylim([-100000000000000000000000000,1000000000000000000000000000])
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed samples')
plt.show()



#With standardization
#calculating mean center
mean_vec=np.mean(data,axis=0)
mean_vec.shape


covariance_matrix = np.divide(((data-mean_vec).T.dot(data-mean_vec).T),len(data))
covariance_matrix
print('Covariance Matrix \n%s'%covariance_matrix)




#1. Total Variance of original variables
np.diagonal(covariance_matrix)
print('Covariance Matrix \n%s'%covariance_matrix)


#performing eigen value decomposition
eigen_vals, eigen_vecs = np.linalg.eig(np.cov(data.T))
print('Eigen Vectors \n %s' %eigen_vecs)
print('Eigen Values \n %s' %eigen_vals)
eigen_vals.shape



# create a list of tuple (eigenvalue, eigenvector) 
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

np.diagonal(eigen_vecs)

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda x: x[0], reverse=True)
eigen_pairs



# checking the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eigen_pairs:
    eigen=i[0]
    print eigen


#explained variance    
#4. It can be noted from the cum_var_exp array that we need 4 prinicpal components to keep 75% of the variance
total = sum(eigen_vals)
var_exp = [(i / total)*100 for i in sorted(eigen_vals, reverse=True)]
sum(var_exp)
cum_var_exp = np.cumsum(var_exp)
cum_var_exp

#plotting prinicpal components
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(7, 7))

    plt.bar(range(3), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
   
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()



#projection matrix:selecting first two pc
matrix_w = np.hstack((eigen_pairs[0][1].reshape(49,1),
                      eigen_pairs[1][1].reshape(49,1)))

print("projection matrix:\n%s"%matrix_w) 

# 2.) covariance of first PC and second PC
cov_pc = np.cov(matrix_w[0:49,0],matrix_w[0:49,1])
cov_pc

# projecting to new space
Y = data.dot(matrix_w)
Y.shape

#p3.) plotting scores between first and second pc  in red AND blue
plt.plot(Y[0:20,0],Y[0:20,1], 'o', markersize=7, color='red', alpha=0.5, label='class1')
plt.plot(Y[20:40,0],Y[20:40,1], '^', markersize=7, color='blue', alpha=0.5, label='class2')
plt.xlim([-100000000000000000000000000,1000000000000000000000000000])
plt.ylim([-100000000000000000000000000,1000000000000000000000000000])
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed samples')
plt.show()