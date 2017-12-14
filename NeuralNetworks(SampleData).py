# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 16:38:05 2017
@author: vikhyat
"""
#Neural networks with parameters on sample data
import numpy as np
from matplotlib import pyplot as plt

def NN(m1,m2,w1,w2,b):
    z=m1*w1+m2*w2+b
    return sigmoid(z)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def calculate_cost(pred,target):
    return (((pred-target)**2)/2)

#---------------------------------#
#Randomly assigning weights
w1=np.random.uniform(0,1)
w2=np.random.uniform(0,1)
w3=np.random.uniform(0,1)
w4=np.random.uniform(0,1)
w5=np.random.uniform(0,1)
w6=np.random.uniform(0,1)
w7=np.random.uniform(0,1)
w8=np.random.uniform(0,1)
b11=np.random.uniform(0,1)
b12=np.random.uniform(0,1)
b21=np.random.uniform(0,1)
b22=np.random.uniform(0,1)
iterations=[]
tc=[]
theta1H=[]
theta2H=[]

for i in range(1000):
    #Initial inputs
    out_h1= NN(0.05,0.1,w1,w2,b11)
    out_h2= NN(0.05,0.1,w3,w4,b12)
    out_o1=NN(out_h1,out_h1,w5,w6,b21)
    out_o2=NN(out_h1,out_h2,w7,w8,b22)
    #Calculating cost as per the target value
    total_cost=calculate_cost(out_o1,0.1)+calculate_cost(out_o2,0.99)
    iterations.append(i+1)
    tc.append(total_cost)
    #Outer layer delta
    d1_3= ((out_o1-0.01)*(out_o1*(1-out_o1)))
    d2_3= ((out_o2-0.99)*(out_o2*(1-out_o2)))
    d_3 = np.matrix([d1_3,d2_3]).reshape(2,1)
    a_2 = np.matrix([1,out_h1,out_h2])
    Delta_3 = np.dot(d_3,a_2)
    Delta_3
    #Storing into history for plotting purposes
    theta1H.append([w1,w2,w3,w4,b11,b12])
    theta2H.append([w5,w6,w7,w8,b21,b22])
    #Making changes to the outer layer weights with learning rate of 0.5
    w5-= Delta_3[0,0]/2
    w6-= Delta_3[0,1]/2
    w7-= Delta_3[0,2]/2
    w8-= Delta_3[1,0]/2
    b21-=Delta_3[1,1]/2
    b22-=Delta_3[1,2]/2
    #Hidden layer delta gradient
    d1_z1= ((out_o1-0.01)*(out_o1*(1-out_o1))*w5*(out_h1*(1-out_h1))) 
    d2_z1= ((out_o2-0.99)*(out_o2*(1-out_o2))*w7*(out_h1*(1-out_h1)))
    d_z1=d1_z1+d2_z1
    d1_z2= ((out_o1-0.01)*(out_o1*(1-out_o1))*w6*(out_h2*(1-out_h2)))
    d2_z2= ((out_o2-0.99)*(out_o2*(1-out_o2))*w8*(out_h2*(1-out_h2)))
    d_z2=d1_z2+d2_z2
    
    d_2 = np.matrix([d_z1,d_z2]).reshape(2,1)
    a_1 = np.matrix([1,out_h1,out_h2])
    Delta_2 = np.dot(d_2,a_1)
    Delta_2
    #Updating weights
    w1-= Delta_2[0,0]/2
    w2-= Delta_2[0,1]/2
    w3-= Delta_2[0,2]/2
    w4-= Delta_2[1,0]/2
    b11-=Delta_2[1,1]/2
    b12-=Delta_2[1,2]/2
    print(total_cost)
   
plt.plot(iterations,tc)
plt.show()


theta1H_1 = [item[0] for item in theta1H]
theta1H_2 = [item[1] for item in theta1H]
theta1H_3 = [item[2] for item in theta1H]
theta1H_4 = [item[3] for item in theta1H]
theta1H_5 = [item[4] for item in theta1H]
theta1H_6 = [item[5] for item in theta1H]

theta2H_1 = [item[0] for item in theta2H]
theta2H_2 = [item[1] for item in theta2H]
theta2H_3 = [item[2] for item in theta2H]
theta2H_4 = [item[3] for item in theta2H]
theta2H_5 = [item[4] for item in theta2H]
theta2H_6 = [item[5] for item in theta2H]
#Total cost graph
plt.plot(iterations,theta1H[0])
plt.show()
#Weights graph
plt.plot(iterations,theta1H_1)
plt.plot(iterations,theta1H_2)
plt.plot(iterations,theta1H_3)
plt.plot(iterations,theta1H_4)
plt.plot(iterations,theta1H_5)
plt.plot(iterations,theta1H_6)
plt.show()

plt.plot(iterations,theta2H_1)
plt.plot(iterations,theta2H_2)
plt.plot(iterations,theta2H_3)
plt.plot(iterations,theta2H_4)
plt.plot(iterations,theta2H_5)
plt.plot(iterations,theta2H_6)
plt.show()
    



    

    
    
   
   

