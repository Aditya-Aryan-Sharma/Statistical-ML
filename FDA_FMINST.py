import pandas as pd
import numpy as np
import ssl
import matplotlib.pyplot as plt
from numpy.linalg import eig
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
        
ssl._create_default_https_context = ssl._create_unverified_context
train_set = pd.read_csv(r'C:\pytorch_datasets\Fmnist\\fashion-mnist_train.csv')
test_set = pd.read_csv(r'C:\pytorch_datasets\Fmnist\\fashion-mnist_test.csv')
label = {0:"T-shirt/top",1 :"Trouser",2 :"Pullover",3 :"Dress",4 :"Coat",5 :"Sandal",6 :"Shirt",7 :"Sneaker",8 :"Bag",9 :"Ankle boot"}
train_x = np.array(train_set.iloc[:,1:]).reshape(train_set.shape[0],28,28)
train_y = np.array(train_set.iloc[:,0])
test_x = np.array(test_set.iloc[:,1:]).reshape(test_set.shape[0],28,28)
test_y = np.array(test_set.iloc[:,0])
labels = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0} 
dictionary = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0} 

St = np.cov(np.transpose(train_x.reshape(-1,784))) * (train_x.shape[0] - 1)
x_train = [[],[],[],[],[],[],[],[],[],[]]  
Sw = 0 
for i in range(train_x.shape[0]):
    x_train[train_y[i]].append(train_x[i])   
for j in range(len(x_train)):
    x_train[j] = np.array(x_train[j])
    if i == 0:    
        Sw = np.cov(np.transpose(x_train[j].reshape(-1,784))) * (x_train[j].shape[0] -1)
    else:
        Sw = Sw + np.cov(np.transpose(x_train[j].reshape(-1,784))) * (x_train[j].shape[0] - 1)
Sb = St - Sw                 

dimension = 1
A = np.dot(np.linalg.inv(Sw),Sb)
values, vectors = eig(A)
values = values.real
vectors = vectors.real
eiglist = [(values[i],vectors[:,i]) for i in range(len(values))]
eiglist = sorted(eiglist, key = lambda x: x[0], reverse = True)
W = np.array([eiglist[i][1] for i in range(dimension)])  
print("W = ", W)  
Y = np.dot(W, np.transpose(train_x).reshape(784,-1))

classifier = LDA()
classifier.fit(np.transpose(Y) , train_y) 
LDA(n_components=None, priors=None, shrinkage=None, solver='svd',store_covariance=False, tol=0.0001)
test = np.array(W.dot(np.array(test_x).transpose().reshape(784,-1))).transpose()
predicted_y = classifier.predict(test)
accuracy = accuracy_score(test_y,predicted_y)*100 
print("Accuracy = " , accuracy, "%")
for i in range(len(test_y)):
    labels[test_y[i]] += 1
    if classifier.predict(test[i].reshape(-1,dimension))[0] == test_y[i]:
        dictionary[test_y[i]] += 1
for i in range(len(labels)):
    print("class ",i," accuracy = ",(dictionary[i]/labels[i])*100," %")   
