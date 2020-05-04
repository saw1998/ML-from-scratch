
import numpy as np


class NeuralNetwork:        
    def __init__(self):
        self.test_data = None
        self.train_data=None
        self.no_of_layer=0
        self.layer=[]
        self.activation_function=[]
        self.learning_rate=None
        self.epochs=None
        self.loss_function='categorical cross entropy loss'
        self.weights=[]
        self.delta=[]
        self.error=[]
        self.batch_size=10
        self.count=0
        self.error1=0
        self.train_accuracy=[]
        self.test_accuracy=[]
        
        sampl = np.random.uniform(low=0.5, high=13.3, size=(50,))
    def add_layer(self,no_of_neuron,activation_function=None,output_layer=False):
        np.random.seed(0)                                                           #to get same output while testing
        if(output_layer):
            self.activation_function.append(activation_function)   #adding activation function to current layer
            self.delta.append(np.ones(no_of_neuron))   #creating space for deltas
            self.weights.append(0.1*np.random.uniform(-1,1,size=(no_of_neuron,len(self.layer[self.no_of_layer-1])))) #random initialization of weights
            self.layer.append(np.ones(no_of_neuron))
            self.no_of_layer+=1
            return
        
        elif(self.no_of_layer!=0):
            self.activation_function.append(activation_function)   #adding activation function to current layer
            self.weights.append(0.1*np.random.uniform(-1,1,size=(no_of_neuron,len(self.layer[self.no_of_layer-1])))) #random initialization of weights                
            self.delta.append(np.ones(no_of_neuron))   #creating space for deltas
        else:
            self.activation_function.append(None)
            self.weights.append(None)
            self.delta.append(None)
            
        self.no_of_layer+=1    
        self.layer.append(np.ones(no_of_neuron+1))      # adding neuron to model
        return
        
    def forward_propogation(self,data):
        self.layer[0] = np.append(1,data)
        act_fun=None
        for i in range(1,self.no_of_layer-1):
            act_fun=self.find_function(self.activation_function[i])
            self.layer[i]=np.append(1,act_fun(np.dot(self.weights[i],self.layer[i-1])))
        i=self.no_of_layer-1
        act_fun=self.find_function(self.activation_function[i])
        self.layer[i]=act_fun(np.dot(self.weights[i],self.layer[i-1]))
        return
    
    def find_function(self,string):
                      
        def sigmoid(arr,derivative=False):
            if(derivative):
#                 sig=sigmoid(arr)                            #already calculated values were stored
                return(arr*(1-arr))
            return(1/(1+np.exp(-arr)))
        
        def relu(arr,derivative=False):
            if(derivative):
                return(np.array([int(i>0) for i in arr]))
            return(np.array([max(0,i) for i in arr]))
                      
        def softmax(arr,derivative=False):
            if(derivative):
#                 sof=softmax(arr)                              #already calculated values were stored
                return(arr-arr*arr)
            return(np.exp(arr) / sum(np.exp(arr)))
                
        if(string=='sigmoid'):
            return(sigmoid)
        if(string=='relu'):
            return(relu)
        if(string=='softmax'):
            return(softmax)
        return(None)
                      
    def cross_entropy_loss(self,actual,output):                             #on single training example

        self.error.append(np.sum(-actual*np.log2(output)-(1-actual)*np.log2(1-output)))
        return
        
    def Backward_propogation(self,actual,nth_layer=-1):
        if(nth_layer==0):
            return
        if(nth_layer==-1):
            actual=np.array([int(i==actual-1) for i in range(len(self.layer[self.no_of_layer-1]))])  #todo uncomment it
            nth_layer=self.no_of_layer
            output=self.layer[self.no_of_layer-1]                                                                   #####checked#####
            self.cross_entropy_loss(actual,output)
            self.delta[nth_layer-1]+=(output-actual)  #considering softmax activation function and cross-entropy loss function
            return

        act_fun=find_function(self.activation_function[nth_layer-1])
        theta_dash=np.array(act_fun(self.layer[nth_layer],derivative=True))
        summation=np.dot(self.weight[nth_layer-1].transpose(),self.delta[nth_layer-1])
        self.delta[nth_layer]+=summation*theta_dash
        return(self.Backward_propogation(nth_layer-1))
    
    def clear_delta(self):
        for i in range(1,self.no_of_layer):
            self.delta[i]*=0
        return
    
    def train(self,training_input,training_output,testing_input,testing_output,epoc,learning_rate,batch_size=10):

        training_input=np.array(training_input)
        training_output=np.array(training_output)
        self.learning_rate=learning_rate                                                            # for trainging the model
        self.batch_size=batch_size
        self.clear_delta()
        for j in range(epoc):
            if(j%10==0):
                self.train_accuracy.append(self.test(training_input,training_output))
                self.test_accuracy.append(self.test(testing_input,testing_output))
            r=len(training_input)
            for i in range(r):
                self.count+=1
                self.forward_propogation(training_input[i])
#                 print("training_output=",trainging_output[i])
                self.Backward_propogation(training_output[i])
                if(self.count%self.batch_size==0):
                    self.update_weights()
                    self.clear_delta()
        return
    
    def update_weights(self):                                                               #for updating the weights
        for i in range(self.no_of_layer-1,0,-1):
#             print(self.delta[i])
#             print(self.layer[i-1])
            difference=self.learning_rate*np.outer(self.delta[i],self.layer[i-1])
#             print(difference)
#             print(self.weights[i])
            self.weights[i]=self.weights[i]-difference
        return
    
    def test(self,data,output):
        # result=np.zeros(len(data))
        result=0
        acc=0
        for i in range(len(data)):                                                  #for testing the model
            self.forward_propogation(data[i])
            result=self.layer[self.no_of_layer-1].argmax(axis=0)+1
            if(result==output[i]):
                acc+=1
        return(acc*100/(len(data)))

        

            


def main():

    print("Part 1A:")
    file=open('../data/train_data.txt')
    data=[]
    temp=file.readline().split()
    # temp_row=[]
    while(temp!=[]):
        temp_row=[]
        for i in range(len(temp)):                                          #preparing data
            temp_row.append(float(temp[i]))
        data.append(temp_row)
    #     print(data)
        temp=file.readline().split()
    data=np.array(data)
    X_train=data[:,0:7]
    y_train=data[:,7]



    file=open('../data/test_data.txt')
    data=[]
    temp=file.readline().split()
    # temp_row=[]
    while(temp!=[]):
        temp_row=[]
        for i in range(len(temp)):                                          #preparing test data
            temp_row.append(float(temp[i]))
        data.append(temp_row)
    #     print(data)
        temp=file.readline().split()
    data=np.array(data)
    X_test=data[:,0:7]
    y_test=data[:,7]


    model=NeuralNetwork()                                                   #creating instance of NeuralNetwork
    model.add_layer(7)                                      
    model.add_layer(32,'sigmoid')                                                   #adding layer
    model.add_layer(3,"softmax",output_layer=True)

    model.train(X_train,y_train,X_test,y_test,epoc=200,learning_rate=0.01,batch_size=10)                             #trainging
    _sum=0
    error_list=[]
    for i in range(1,len(model.error)):
        _sum+=model.error[i]
        if(i%(len(X_train)*10)==0):                                               #preparing errorlist for plotting after every 10 epocs
            error_list.append(_sum/(len(X_train)*10))
            _sum=0





    import matplotlib.pyplot as plt

    plt.figure()
    l=10*np.arange(len(error_list))                                            #plotting the errors
    err=error_list
    plt.plot(l,err)
    plt.xlabel('epoc', fontsize=18)
    plt.ylabel('error', fontsize=16)
    plt.show()

    plt.figure()
    l1=10*np.arange(len(model.train_accuracy))                                            #plotting the errors
    acc1=model.train_accuracy
    l2=10*np.arange(len(model.test_accuracy))                                            #plotting the errors
    acc2=model.test_accuracy
    plt.plot(l1,acc1,'r',label="accuravy on training data")
    plt.plot(l2,acc2,'b',label="accuravy on testing data")
    plt.xlabel('epoc', fontsize=18)
    plt.ylabel('accuracy', fontsize=16)
    plt.legend()
    plt.show()


    accuracy=model.test(X_train,y_train)                                                    #testing on traing data                                             #printing accuracy
    print("Final accuracy on training data is ",accuracy,"%")

    accuracy=model.test(X_test,y_test)                                                      #testing on test data
    print("Final accuracy on testing data is ",accuracy,"%")


############################################################### part b start ############################################################

    print("Part 1B:")

    file=open('../data/train_data.txt')
    data=[]
    temp=file.readline().split()
    # temp_row=[]
    while(temp!=[]):
        temp_row=[]
        for i in range(len(temp)):                                          #preparing data
            temp_row.append(float(temp[i]))
        data.append(temp_row)
    #     print(data)
        temp=file.readline().split()
    data=np.array(data)
    X_train=data[:,0:7]
    y_train=data[:,7]



    file=open('../data/test_data.txt')
    data=[]
    temp=file.readline().split()
    # temp_row=[]
    while(temp!=[]):
        temp_row=[]
        for i in range(len(temp)):                                          #preparing test data
            temp_row.append(float(temp[i]))
        data.append(temp_row)
    #     print(data)
        temp=file.readline().split()
    data=np.array(data)
    X_test=data[:,0:7]
    y_test=data[:,7]


    model=NeuralNetwork()                                                   #creating instance of NeuralNetwork
    model.add_layer(7)                                      
    model.add_layer(64,'relu')                              
    model.add_layer(32,'relu')                                                   #adding layer
    model.add_layer(3,"softmax",output_layer=True)

    model.train(X_train,y_train,X_test,y_test,epoc=200,learning_rate=0.01,batch_size=10)                             #trainging
    _sum=0
    error_list=[]
    for i in range(1,len(model.error)):
        _sum+=model.error[i]
        if(i%(len(X_train)*10)==0):                                               #preparing errorlist for plotting after every 10 epocs
            error_list.append(_sum/(len(X_train)*10))
            _sum=0





    import matplotlib.pyplot as plt

    plt.figure()
    l=10*np.arange(len(error_list))                                            #plotting the errors
    err=error_list
    plt.plot(l,err)
    plt.xlabel('epoc', fontsize=18)
    plt.ylabel('error', fontsize=16)
    plt.show()

    plt.figure()
    l1=10*np.arange(len(model.train_accuracy))                                            #plotting the errors
    acc1=model.train_accuracy
    l2=10*np.arange(len(model.test_accuracy))                                            #plotting the errors
    acc2=model.test_accuracy
    plt.plot(l1,acc1,'r',label="accuravy on training data")
    plt.plot(l2,acc2,'b',label="accuravy on testing data")
    plt.xlabel('epoc', fontsize=18)
    plt.ylabel('accuracy', fontsize=16)
    plt.legend()
    plt.show()


    accuracy=model.test(X_train,y_train)                                                    #testing on traing data                                             #printing accuracy
    print("Final accuracy on training data is ",accuracy,"%")

    accuracy=model.test(X_test,y_test)                                                      #testing on test data
    print("Final accuracy on testing data is ",accuracy,"%")




if __name__ == '__main__':
    main()