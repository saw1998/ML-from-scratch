
import numpy as np
file=open('../data/seeds_dataset.txt')

data=[]
temp=file.readline().split()
while(temp!=[]):
    temp_row=[]
    for i in range(len(temp)):
        temp_row.append(float(temp[i]))
    data.append(temp_row)
#     print(data)
    temp=file.readline().split()

data=np.array(data)

output=[i[7] for i in data]

mean=np.zeros(8)
for row in data:
    mean+=row

mean/=len(data)

std=np.zeros(8)
for row in data:
    std+=(row-mean)**2

std/=len(data)
std=np.sqrt(std)

for i in range(len(data)):
    data[i]=(data[i]-mean)/std

for i in range(len(data)):
    data[i][7]=output[i]


x = np.random.rand(100, 5)
np.random.seed(7)
np.random.shuffle(data)
np.random.shuffle(data)
np.random.shuffle(data)
N=int(0.8*len(data))

train, test = data[:N,:], data[N:,:]

np.savetxt("../data/train_data.txt",train)

np.savetxt("../data/test_data.txt",test)



