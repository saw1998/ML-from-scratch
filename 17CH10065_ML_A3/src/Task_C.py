#!/usr/bin/env python
import numpy as np
import pandas as pd

def distance(centroids,data,K):
#     print(centroids)
    dis=np.ones(K)
    for i in range(K):								#calculating distances between centroid and data poing
        dis[i]=np.dot(centroids[i],data)
        dinominator=np.sqrt(np.dot(centroids[i],centroids[i])*np.dot(data,data))
        dis[i]=dis[i]/dinominator
    return(np.exp(-1*dis))




def K_mean(data,K=8,iterations=200):
#     iterations=200
    centroids=np.random.rand(K,len(data[0]))				#K_mean algorithm
    belongs_to=np.ones(len(data))
    SSE=np.zeros(iterations)

    for itr in range(iterations):
        for i in range(len(data)):
            distances=distance(centroids,data[i],K)

            SSE[itr]+=(distances.sum())

            min_dist_index=0
            min_dist=distances[0]
            for j in range(1,K):
                if(distances[j]<min_dist):
                    min_dist=distances[j]
                    min_dist_index=j
    #                 print(j,end=" ")

            belongs_to[i]=min_dist_index


    #     print("hello")
        centroids[:]=0
        count=np.zeros(K)
        for i in range(len(belongs_to)):
            centroids[int(belongs_to[i])]+=data[i]
            count[int(belongs_to[i])]+=1

        for i in range(K):
            if(count[i]!=0):
                centroids[i]/=count[i]
            else:
                print("*",end="")
        print("#",end="")
        
        cluster=[[],[],[],[],[],[],[],[]]
        for i in range(len(belongs_to)):
            cluster[int(belongs_to[i])].append(i)
        
#         sorted_cluster=sorted(cluster)
        
    return(cluster,centroids,SSE)





def main():

    df=pd.read_csv("../data/modified.csv")			#loading file
    print(df.head())
    np_df=df.drop('Unnamed: 0',axis=1)
    data=np.array(np_df)

    clusters,centroid,SSE=K_mean(data)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.arange(200),SSE)
    plt.show()



    sorted_cluster=[]
    for i in clusters:
        sorted_cluster.append(sorted(i))                              #########pass############

    f=open("../clusters/kmeans.txt",'w')				#storing clusters in file
    for i in sorted_cluster:
        for j in range(len(i)):
            f.write(str(i[j]))
            if(j!=len(i)-1):
                f.write(",")
        f.write("\n")
    f.close()
    print()


if __name__=="__main__":
    main()


