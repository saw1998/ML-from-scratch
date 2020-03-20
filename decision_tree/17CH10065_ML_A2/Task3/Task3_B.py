#!/usr/bin/env python


import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

df=pd.read_csv("../dataset/dataset_B.csv")
X=df.drop(['quality'],axis=1)
y=df['quality']



print("############## using sklearn model #####################")

print()
model=DecisionTreeClassifier(criterion='entropy',min_samples_split=10)
model.fit(X,y)


y_pred=model.predict(X)
print("accuracy_score=",accuracy_score(y,y_pred))
print()
print("precision_score=",precision_score(y,y_pred,average='macro'))
print()
print("recall_score=",recall_score(y,y_pred,average='macro'))
print()
print("f1_score=",f1_score(y,y_pred,average='macro'))
print()
print("Confusion Matrix")
print(confusion_matrix(y,y_pred))



######################## Tast3-C subpart ###############################
"""
from sklearn.model_selection import cross_val_score

model=DecisionTreeClassifier(criterion='entropy',min_samples_split=10)
print(cross_val_score(model,X,y,cv=3, ))

"""


