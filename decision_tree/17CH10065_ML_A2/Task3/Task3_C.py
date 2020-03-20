#!/usr/bin/env python

from Task3_A import *
from sklearn.tree import DecisionTreeClassifier					#importing necessary files
from sklearn.metrics import accuracy_score,precision_score,recall_score


dataf=pd.read_csv("../dataset/dataset_B.csv")					#importing data


test1=dataf.iloc[0:534][:]
test2=dataf.iloc[534:1067][:]
test3=dataf.iloc[1067:][:]

train1 = pd.concat([test2, test3])						#dividing data in three parts
train2 = pd.concat([test1, test3])
train3 = pd.concat([test1, test2])

train=[train1,train2,train3]
test=[test1,test2,test3]


accuracy=np.zeros(3)
precision=np.zeros(3)
recall=np.zeros(3)


y_test=[]



print("Training on dataset")
for i in range(3):							# training the model using model defined in Task3-B.py
    head=node()
    fit_tree(head,train[i],'quality')

    print("prediction the output on testing dataset",i)
    y_pred=np.ones(len(test[i]))
    for j in range(len(test[i])):
        y_pred[j]=classify(head,test[i].iloc[j])
    #     print((df.iloc[item]))
    y_test.append(test[i]['quality'])
    
    accuracy[i]=accuracy_score(y_test[i],y_pred)
    precision[i]=precision_score(y_test[i],y_pred,average='macro')
    recall[i]=recall_score(y_test[i],y_pred,average='macro')


print()
print("######### without using scikit learn library   #############")			#printing results
print("mean accuracy=",accuracy.mean())
print("mean precision=",precision.mean())
print("mean recall=",recall.mean())






accuracy=np.zeros(3)
precision=np.zeros(3)							#preparing data
recall=np.zeros(3)
y=[]
for i in range(3):
    y.append(train[i]['quality'])
    train[i].drop(['quality'],axis=1)



for i in range(3):
    model=DecisionTreeClassifier(criterion='entropy',min_samples_split=10)	#training, testing and storing the result using scikit learn library
    model.fit(train[i],y[i])
    
    y_pred=model.predict(test[i])
    
    accuracy[i]=accuracy_score(y_test[i],y_pred)
    precision[i]=precision_score(y_test[i],y_pred,average='macro')
    recall[i]=recall_score(y_test[i],y_pred,average='macro')

print()
print("######### using scikit learn library   #############")			#printing result
print("mean accuracy=",accuracy.mean())
print("mean precision=",precision.mean())
print("mean recall=",recall.mean())




