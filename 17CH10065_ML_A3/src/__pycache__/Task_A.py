#!/usr/bin/env python

import numpy as np
import pandas as pd

df=pd.read_csv("../data/AllBooks_baseline_DTM_Labelled.csv")				#reading the data

print(df.head())

for i in range(len(df)):
    df.at[i,'Unnamed: 0']=df.iloc[i][0].rsplit("_",1)[0]
    																			#manupulating the data
df.drop(df.index[13],axis=0,inplace=True)

print(df.head(20))
df.reset_index(drop=True,inplace=True)
df.to_csv("../data/modified.csv",index=False)									#saving the data in modified.csv
	


########### TF-IDF ######### using sklearn ################

class_labels=np.array(df['Unnamed: 0'])
df=df.drop(['Unnamed: 0'],axis=1)

from sklearn.feature_extraction.text import TfidfTransformer
from numpy import linalg as LA

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)                                           # finding tfidf
tfidf_transformer.fit(df)

tf_idf_vec=tfidf_transformer.transform(df)

feature_names = np.array(df.columns)


rows=[]
for i in range(len(df)):
    document_vec=tf_idf_vec[i]
    data = pd.DataFrame(document_vec.T.todense(), index=feature_names, columns=[class_labels[i]])			# storing calculated values
    data=data/LA.norm(document_vec.T.todense())
    rows.append(data.T)
    print('#',end='')
data_frame_Tfidf=pd.concat(rows)

print(data_frame_Tfidf.head())

data_frame_Tfidf.to_csv("../data/data_frame_Tfidf.csv")