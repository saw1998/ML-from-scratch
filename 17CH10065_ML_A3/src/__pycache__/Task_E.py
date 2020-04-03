#!/usr/bin/env python
import numpy as np
import pandas as pd


def entropy_class(classes,class_using_index):
    for i in class_using_index:
        classes[i]+=1
    entropy=0.0
    total_data_point=len(class_using_index)
    for i in classes:
        temp=classes[i]/total_data_point
        if(temp!=0):
            entropy+=-1*temp*np.log2(temp)
    return(entropy)


def entropy_clusters(clusters,data):
    total_data_point=len(data)
    entropy=0.0
    for i in range(len(clusters)):
        temp=len(clusters[i])/total_data_point
        if(temp!=0):
            entropy-=temp*np.log2(temp)
    return(entropy)



def conditional_entropy(data,cluster):
    total_data_point=len(data)
    points_in_cluster=len(cluster)
    class_in_cluster={'Buddhism':0,"TaoTeChing":0,"Upanishad":0,"YogaSutra":0,"BookOfProverb":0,"BookOfEcclesiastes":0,"BookOfEcclesiastes":0,"BookOfEccleasiasticus":0,"BookOfWisdom":0}
    for point in cluster:
        class_in_cluster[data[int(point)]]+=1

    cond_entropy=0.0
    for i in class_in_cluster:
        temp=class_in_cluster[i]/points_in_cluster
        if(temp!=0):
            cond_entropy+= temp*np.log2(temp)
    cond_entropy*= -1 * points_in_cluster / total_data_point
    return(cond_entropy)



def mutual_information(data,clusters,class_entropy):
    total_conditional_entropy=0.0
    for cluster in clusters:
        temp=conditional_entropy(data,cluster)
        total_conditional_entropy += temp
    #cluster_entropy=entropy_clusters(clusters,data)
    return(class_entropy - total_conditional_entropy)



def NMI(clusters,data,class_entropy):
    return(2*mutual_information(data,clusters,class_entropy)/(entropy_clusters(clusters,data)+class_entropy))           

def main():

    df=pd.read_csv('../data/modified.csv')                                                                            #main function
    class_using_index=np.array([df.iloc[i][0] for i in range(len(df))])


    files=['agglomerative.txt','kmeans.txt','agglomerative_reduced.txt','kmeans_reduced.txt']
    methods=['Aggolmerization','K-means','Agglometization on reduced data','K-means on reduced data']
    classes={'Buddhism':0,"TaoTeChing":0,"Upanishad":0,"YogaSutra":0,"BookOfProverb":0,"BookOfEcclesiastes":0,"BookOfEcclesiastes":0,"BookOfEccleasiasticus":0,"BookOfWisdom":0}


    class_entropy=entropy_class(classes,class_using_index)
    for f in range(len(files)):
        clusters=[]
        path="../clusters/"+files[f]                                                             #pass
        with open(path, "r") as filestream:
            for line in filestream:
                line1=line.split("\n")
                del(line1[len(line1)-1])
                clusters.append(line1[0].split(sep=','))
        
        
        print("Normalized Mutual Information using",methods[f],"is",NMI(clusters,class_using_index,class_entropy))
    
        


if __name__=="__main__":
    main()

