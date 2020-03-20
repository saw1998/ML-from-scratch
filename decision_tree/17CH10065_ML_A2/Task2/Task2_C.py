#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix



dataf=pd.read_csv("../dataset/dataset_A.csv")
dataf.head()

test1=dataf.iloc[0:534][:]			##preparing dataset
test2=dataf.iloc[534:1067][:]
test3=dataf.iloc[1067:][:]

train1 = pd.concat([test2, test3])
train2 = pd.concat([test1, test3])
train3 = pd.concat([test1, test2])

train=[train1,train2,train3]
test=[test1,test2,test3]



accuracy=np.zeros(3)
precision=np.zeros(3)
recall=np.zeros(3)



y=[]
y_test=[]


thetas=np.random.randn(12)				 #initialization and data manupulation
for i in range(3):
   
#     thetas=np.random.randn(12)
    y.append(train[i]['quality'])
    del train[i]['quality']
    train[i].insert(0,'fixed',np.ones(y[i].size))

    iteration=50
    alpha=0.1
    m=y[i].size

    one=np.ones(m)
    J_theta=np.ones(iteration)


    completed=0
    for itr in range(iteration):

        if(((itr*100)//iteration) > completed):
            print('#',end='')
            completed=itr//iteration
            
        h_theta=np.ones(y[i].size)
        difference=np.ones(y[i].size)                                                        #difference = h_theta(i) minus y(i)

        for k in range(m):                                                                # m training example
            theta_transpose_x=np.dot(thetas,train[i].iloc[k])
            h_theta[k]=1/(1+np.exp(-1*theta_transpose_x))
        difference=h_theta - y[i]

        J_theta[itr]=(-1/m)*( np.dot(y[i],np.log(h_theta)) + np.dot((1-y[i]),np.log(1-h_theta)) )     # J_theta at each iteration

        for j in range(thetas.size):                                                      #training
            summation = np.dot(difference,one*train[i].iloc[:,j])
            thetas[j]=thetas[j]-(alpha/m)*summation
            
    ##checking convergence
    
    plt.figure()
    plt.plot(range(iteration),J_theta)
    plt.xlabel('iteration')
    plt.ylabel('Cost')
    plt.title('Cost v/s iteration')
    plt.show()

    
    ##predicting
    
    y_test.append(test[i]['quality'])
    del test[i]['quality']
    test[i].insert(0,'fixed',np.ones(y_test[i].size))
    
    y_pred=np.ones(len(test[i]))
    for j in range(len(test[i])):
        h=1/(1+np.exp(-1*np.dot(thetas,test[i].iloc[j])))
    if(h > 0.5):
        y_pred[j]=1
    elif(h <= 0.5):
        y_pred[j]=0
        
    
    # storing results
    acc=0
    for j in range(len(test[i])):
        h=1/(1+np.exp(-1*np.dot(thetas,test[i].iloc[j])))
        if(h > 0.5 and y_test[i].iloc[j]==1):
            acc=acc+1
        elif(h <= 0.5 and y_test[i].iloc[j]==0):
            acc=acc+1
    accuracy[i]=(acc/(len(test[i])))
#     accuracy[i]=accuracy_score(y_test[i],y_pred)
    precision[i]=precision_score(y_test[i],y_pred)
    recall[i]=recall_score(y_test[i],y_pred)
            
            
############ printing result  ###########################

print()
print("######### without using scikit learn library   #############")
print("mean accuracy=",accuracy.mean())
print("mean precision=",precision.mean())
print("mean recall=",recall.mean())
print("#############################################################")

#using scikit learn model


from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score



for i in range(3):
    del train[i]['fixed']
for i in range(3):
    del test[i]['fixed']

accuracy=np.zeros(3)
precision=np.zeros(3)
recall=np.zeros(3)
    


for i in range(3):
    logistic_regression = LogisticRegression()
    #using sklearn 
    logistic_regression.fit(train[i],y[i])
    LogisticRegression(solver='saga')
    y_pred = logistic_regression.predict(test[i])
    
    accuracy[i]=accuracy_score(y_test[i],y_pred)
    precision[i]=precision_score(y_test[i],y_pred)
    recall[i]=recall_score(y_test[i],y_pred)

############ printing result  ###########################

print()
print("######### using scikit learn library   #############")
print("mean accuracy=",accuracy.mean())
print("mean precision=",precision.mean())
print("mean recall=",recall.mean())
print("#############################################################")



