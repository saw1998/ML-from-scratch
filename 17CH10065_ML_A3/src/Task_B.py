#!/usr/bin/env python
import numpy as np
import pandas as pd

def cosine_similarity(data1,data2):
    numerator=np.dot(data1,data2)
    dinominator=np.sqrt(np.dot(data1,data1)*np.dot(data2,data2))                                             #cosine Similiraty
    return(np.exp(-1*numerator/dinominator))



def single_linkage(cluster,distances,index1,index2):
    mini=999999
    for i in cluster[index1]:								#single linkage (minimum distance between two cluster)
        for j in cluster[index2]:
            if(mini>distances[i][j]):
                mini=distances[i][j]
    return(mini)



def make_cluster(cluster,distances,no_of_cluster):
    print("#",end="")
    if(len(cluster)<=no_of_cluster):						#recursively making clusters
        return
    min1=999999
    combine=[0,1]
    for i in range(len(cluster)):
        for j in range(i+1,len(cluster)):
            temp=single_linkage(cluster,distances,i,j)
            if(min1>temp):
#                 print(temp,end="...")
                min1=temp
                combine[0]=i
                combine[1]=j
    cluster[combine[0]]=cluster[combine[0]]+cluster[combine[1]]
    del(cluster[combine[1]])
    make_cluster(cluster,distances,no_of_cluster)
    return





def Agglomerative_Clustering(data,no_of_cluster):
#     data=np.array(df.drop('Unnamed: 0',axis=1))
    distance_matrix=np.array([[0.0 for i in range(len(data))] for i in range(len(data))])
    for i in range(len(data)):
        j=0
        while(j<i):										#preparing distance matrix
            distance_matrix[i][j]=distance_matrix[j][i]=cosine_similarity(data[i],data[j])
            j+=1
    
    cluster=[]
    for i in range(len(data)):
        cluster.append([i])
    make_cluster(cluster,distance_matrix,no_of_cluster)
    
        
    sorted_cluster=sorted(cluster)
    return(cluster)





def main():
    df=pd.read_csv("../data/modified.csv")
    print(df.head())									#main function

    data=np.array(df.drop('Unnamed: 0',axis=1))
    clusters=Agglomerative_Clustering(data,8)

    sorted_cluster=[]
    for i in clusters:
        sorted_cluster.append(sorted(i))                              #########pass############

    print()
    print("Result were printed in file --> agglomerative.txt")
    print()
    f=open("../clusters/agglomerative.txt",'w')
    for i in sorted_cluster:
        for j in range(len(i)):							#writing in file
            f.write(str(i[j]))
            if(j!=len(i)-1):
                f.write(",")
        f.write("\n")
    #     f.write(str(i))
    #     f.write("\n")
    f.close()


if __name__=="__main__":
    main()







# from sklearn.cluster import AgglomerativeClustering
# clustering = AgglomerativeClustering(n_clusters=8,affinity="cosine",linkage='single').fit(X)
# clustering.labels_


