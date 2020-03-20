#!/usr/bin/env python

import numpy as np
import pandas as pd


df = pd.read_csv("../dataset/winequality-red.csv",sep=';')			#loading file
df.head()



df.replace({'quality':[0,1,2,3,4]},0,inplace=True)
df.replace({'quality':[5,6]},1,inplace=True)					#manupulating file
df.replace({'quality':[7,8,9]},2,inplace=True)



for column in df:
    if(column=='quality'):
        break
    mean=df[column].mean()
    sd=df[column].std()
    mn=(df[column].min()-mean)/sd
    mx=(df[column].max()-mean)/sd
    rnge=mx-mn

    for i in range(df[column].size):
        val=(df[column][i]-mean)/sd
        if( val <= mn+(rnge/4)):
            df.at[i,column]=0
        elif(val > mn+(rnge/4) and val <= mn+(rnge/2)):
            df.at[i,column]=1
        elif(val > mn+(rnge/2) and val <= mn+(3*rnge/4)):
            df.at[i,column]=2
        else:
            df.at[i,column]=3



df.to_csv("../dataset/dataset_B.csv",index=False)				#saving file




