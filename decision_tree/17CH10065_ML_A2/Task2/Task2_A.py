#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df=pd.read_csv("../dataset/dataset_A.csv")




#initialization and data manupulation

thetas=np.random.randn(12)
y=df['quality']
del df['quality']
df.insert(0,'fixed',np.ones(y.size))

############### main algorithm ####################

iteration=200
alpha=0.1
m=y.size

one=np.ones(m)
J_theta=np.ones(iteration)

print("Training....")
completed=0
for itr in range(iteration):
    
    h_theta=np.ones(y.size)
    difference=np.ones(y.size)                                                        #difference = h_theta(i) minus y(i)
    
    for i in range(m):                                                                # m training example
        theta_transpose_x=np.dot(thetas,df.loc[i])
        h_theta[i]=1/(1+np.exp(-1*theta_transpose_x))
    difference=h_theta - y
    
    J_theta[itr]=(-1/m)*( np.dot(y,np.log(h_theta)) + np.dot((1-y),np.log(1-h_theta)) )     # J_theta at each iteration
    
    for j in range(thetas.size):                                                      #training
        summation = np.dot(difference,one*df.iloc[:,j])
        thetas[j]=thetas[j]-(alpha/m)*summation
  
##############  plotting ###############

plt.figure()
plt.plot(range(iteration),J_theta)
plt.xlabel('iteration')
plt.ylabel('Cost')
plt.title('Cost v/s iteration')
plt.show()


############## Accuracy ###################


acc=0
for i in range(m):
    h=1/(1+np.exp(-1*np.dot(thetas,df.loc[i])))
    if(h > 0.5 and y[i]==1):
        acc=acc+1
    elif(h <= 0.5 and y[i]==0):
        acc=acc+1
acc=(acc/m)*100



print()
print(acc,'percent data on training-data are correctly classified')




