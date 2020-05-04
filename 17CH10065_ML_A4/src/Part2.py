from sklearn.neural_network import MLPClassifier
import numpy as np

####################################################### Part 2 Specification 1A #########################################

#################### train-data ##########################
print("Part 2 Specification 1A :")

file=open('../data/train_data.txt')

data=[]
temp=file.readline().split()
# temp_row=[]												# reading and storing data
while(temp!=[]):
    temp_row=[]
    for i in range(len(temp)):
        temp_row.append(float(temp[i]))
    data.append(temp_row)
#     print(data)
    temp=file.readline().split()


data=np.array(data)
X=data[:,0:7]
y=data[:,7]
clf = MLPClassifier(solver='lbfgs', alpha=0.01,hidden_layer_sizes=(32),activation='logistic',batch_size=10,max_iter=200)			#trainging on train-set
clf.fit(X, y)																							   					#testing on train data
p=clf.predict(X)
count=0
for i in range(len(X)):
    if(p[i]==y[i]):
        count+=1
accuracy=count/len(X)
       
print("Accuracy on training data is ",accuracy*100,"%")


##################### test-data ################

file=open('../data/test_data.txt')

data=[]
temp=file.readline().split()
# temp_row=[]												# reading and storing data
while(temp!=[]):
    temp_row=[]
    for i in range(len(temp)):
        temp_row.append(float(temp[i]))
    data.append(temp_row)
#     print(data)
    temp=file.readline().split()

data=np.array(data)
X=data[:,0:7]
y=data[:,7]

clf.fit(X, y)																										#testing on train data
p=clf.predict(X)
count=0
for i in range(len(X)):
    if(p[i]==y[i]):
        count+=1
accuracy=count/len(X)
       
print("Accuracy on testing data is ",accuracy*100,"%")


####################################################### Part 2 Specification 1B #########################################


####################### train-data ###########################
print("Part 2 Specification 1A :")

file=open('../data/train_data.txt')

data=[]
temp=file.readline().split()
# temp_row=[]												# reading and storing data
while(temp!=[]):
    temp_row=[]
    for i in range(len(temp)):
        temp_row.append(float(temp[i]))
    data.append(temp_row)
#     print(data)
    temp=file.readline().split()


data=np.array(data)
X=data[:,0:7]
y=data[:,7]
clf = MLPClassifier(solver='lbfgs', alpha=0.01,hidden_layer_sizes=(64,32),activation='relu',batch_size=10,max_iter=200)			#trainging on train-set
clf.fit(X, y)																							   					#testing on train data
p=clf.predict(X)
count=0
for i in range(len(X)):
    if(p[i]==y[i]):
        count+=1
accuracy=count/len(X)
       
print("Accuracy on training data is ",accuracy*100,"%")


######################### test-data ######################

file=open('../data/test_data.txt')

data=[]
temp=file.readline().split()
# temp_row=[]												# reading and storing data
while(temp!=[]):
    temp_row=[]
    for i in range(len(temp)):
        temp_row.append(float(temp[i]))
    data.append(temp_row)
#     print(data)
    temp=file.readline().split()

data=np.array(data)
X=data[:,0:7]
y=data[:,7]

clf.fit(X, y)																										#testing on test data
p=clf.predict(X)
count=0
for i in range(len(X)):
    if(p[i]==y[i]):
        count+=1
accuracy=count/len(X)
       
print("Accuracy on testing data is ",accuracy*100,"%")
