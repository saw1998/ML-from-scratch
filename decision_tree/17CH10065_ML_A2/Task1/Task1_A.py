#!/usr/bin/env python


import numpy as np
import pandas as pd



df = pd.read_csv("../dataset/winequality-red.csv",sep=';')		# loading file


df.replace({'quality':[0,1,2,3,4,5,6]},0,inplace=True)
df.replace({'quality':[7,8]},1,inplace=True)


for column in df:
    mx=df[column].max()
    mn=df[column].min()
#     print(mx,mn)
#     print()								# manupulating file
    for i in range(df[column].size):
        df.at[i,column]=(df[column][i]-mn)/(mx-mn)


df.head()



df.to_csv('../dataset/dataset_A.csv',index = False)				#saving file




