#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


df=pd.read_csv("../dataset/dataset_A.csv")

y=df['quality']
del df['quality']
X=df



logistic_regression = LogisticRegression()
logistic_regression.fit(X,y)
LogisticRegression(solver='saga')
y_pred = logistic_regression.predict(X)
acc = metrics.accuracy_score(y,y_pred)
print("accuracy =",acc*100.0)



