#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




class node:								#defining node object
    def __init__(self,is_leaf=False,function=0,division_by_column='null',output='null'):
        

        self.is_leaf=is_leaf
        self.function=function
        self.division_by_column=division_by_column
        self.output=output
        self.left=None
        self.right=None
        
    def node_function(self,data,column):			# function on node
        if(self.function==1):
            return([data[data[column]<=0.5],data[data[column]>0.5]])
        elif(self.function==2):
            return([data[data[column]<=1.5] , data[data[column]>1.5]])
        elif(self.function==3):
            return([data[data[column]<=2.5], data[data[column]>2.5]])
        
    
    def set_class(self,data,output_col):			# testing the class, leaf node belongs to
        count=np.zeros(3)
        for i in range(len(data.index)):
            count[int(data[i][output_col])]+=1
        if (count.max()==count[0]):
            self.output=0
        elif(count.max()==count[1]):
            self.output=1
        elif(count.max()==count[2]):
            self.output=2
            


def entropy(data , col):				# definition of entropy
    n=0;
    count=np.zeros(4)
    n=0
    for i in range(len(data)):
        count[int(data.iloc[i][col])] += 1
        n+=1
        
    ent=0;
    for i in range(4):
        if(count[i]!=0):
            ent-=(count[i]/n)*np.log2(count[i]/n)
    return(ent)




def fit_tree(parent,data,output_col):                                     #Recursive algorithm
    
    if(len(data)<=10):                                                    #Base case - 1
        parent.is_leaf=True                                               #no of data is <=10
        parent.set_class(data,output_col)
        return
    									   #Base case - 2
    first_data=data.iloc[0][output_col]                                    #all belong to same class
    for i in range(len(data)):
        if(data.iloc[i][output_col]!=first_data):
            break
    if(i==len(data)-1):
    
        parent.is_leaf=True
        parent.output=first_data
        return
    
    
    min_entropy=1000
    min_entropy_column='null'
    N=len(data)
    min_entropy_function='null'
    for column in data:                   # finding minimum entropy column and function on node for subtree
        for f in range(1,4):
            parent.function=f
            [data_left,data_right]=parent.node_function(data,column)
            total_child_entropy=0
            if(len(data_left!=0)):
                total_child_entropy=(len(data_left)/N)*entropy(data_left,output_col)
            if(len(data_right!=0)):
                total_child_entropy+=(len(data_right)/N)*entropy(data_right,output_col)
            
            if(total_child_entropy<min_entropy):
                min_entropy=total_child_entropy
                min_entropy_column=column
                min_entropy_function=f
    
    parent.function=min_entropy_function
    [data_left,data_right]=parent.node_function(data,min_entropy_column)
    parent.left=node()
    parent.right=node()
    parent.division_by_column=min_entropy_column
    fit_tree(parent.left,data_left,output_col)
    fit_tree(parent.right,data_right,output_col)
    return
        
    


df=pd.read_csv("../dataset/dataset_B.csv")



def classify(node,data_point):
    if(node.is_leaf):
        return(node.output)
    d=data_point[str(node.division_by_column)]
    if(node.function==1):
        if(d<=0.5):
            return(classify(node.left,data_point))
        return(classify(node.right,data_point))
    elif(node.function==2):
        if(d<=1.5):
            return(classify(node.left,data_point))
        return(classify(node.right,data_point))
    elif(node.function==3):
        if(d<=2.5):
            return(classify(node.left,data_point))
        return(classify(node.right,data_point))

        

def main():
    ######## fitting data ################
    print("Training on dataset")
    head=node()
    fit_tree(head,df,'quality')


    ############### prediction  ##################
    print("prediction the output on training dataset")
    y_pred=np.ones(len(df))
    for item in range(len(df)):
        y_pred[item]=classify(head,df.iloc[item])
    #     print((df.iloc[item]))
    y=df['quality']
    y=np.array(y)


    ################# Testing  #######################

    print("Testing the percentage of correct output")
    acc=0
    for i in range(len(y_pred)):
        if(y_pred[i]==y[i]):
            acc+=1
            
    print("correctly classified data =",100*acc/len(y),"%")

######################### calling main function  ###############33###33
if __name__=="__main__":
    main()
