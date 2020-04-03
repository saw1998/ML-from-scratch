#!/usr/bin/env python

import pandas as pd
import numpy as np

df=pd.read_csv("../data/modified.csv")

df.drop('Unnamed: 0',inplace=True,axis=1)
df=np.array(df)


from sklearn.decomposition import PCA
pca = PCA(n_components=100)
principalComponents = pca.fit_transform(df)
reduced_data = pd.DataFrame(data = principalComponents)


from Task_C import *
from Task_B import *


K=8
data=np.array(reduced_data)
iterations=200
clusters,centroid,SSE=K_mean(data,K,iterations)


sorted_cluster=[]
for i in clusters:
    sorted_cluster.append(sorted(i))                              #########pass############

f=open("../clusters/kmeans_reduced.txt",'w')
for i in sorted_cluster:
    for j in range(len(i)):
        f.write(str(i[j]))
        if(j!=len(i)-1):
            f.write(",")
    f.write("\n")
f.close()






clusters=Agglomerative_Clustering(data,8)
sorted_cluster=[]
for i in clusters:
    sorted_cluster.append(sorted(i))                              #########pass############

f=open("../clusters/agglomerative_reduced.txt",'w')
for i in sorted_cluster:
    for j in range(len(i)):
        f.write(str(i[j]))
        if(j!=len(i)-1):
            f.write(",")
    f.write("\n")
f.close()


