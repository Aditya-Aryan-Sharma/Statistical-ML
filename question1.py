import pickle
import os
import numpy as np
import ssl
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

ssl._create_default_https_context = ssl._create_unverified_context
batch_metadata = unpickle("C:\pytorch_datasets\cifar-10\\batches.meta")
labels2id = {index : str(label,'utf-8') for index,label in enumerate(batch_metadata[b'label_names'])}
print(labels2id)
batch_1 = unpickle("C:\pytorch_datasets\cifar-10\\data_batch_1")
batch_2 = unpickle("C:\pytorch_datasets\cifar-10\\data_batch_2")
batch_3 = unpickle("C:\pytorch_datasets\cifar-10\\data_batch_3")
batch_4 = unpickle("C:\pytorch_datasets\cifar-10\\data_batch_4")
batch_5 = unpickle("C:\pytorch_datasets\cifar-10\\data_batch_5")
test_batch = unpickle("C:\pytorch_datasets\cifar-10\\test_batch")

train_x = []
train_x.extend(batch_1[b'data'])
train_x.extend(batch_2[b'data'])
train_x.extend(batch_3[b'data'])
train_x.extend(batch_4[b'data'])
train_x.extend(batch_5[b'data'])
train_x = np.array(train_x)
train_y = []
train_y.extend(batch_1[b'labels'])
train_y.extend(batch_2[b'labels'])
train_y.extend(batch_3[b'labels'])
train_y.extend(batch_4[b'labels'])
train_y.extend(batch_5[b'labels'])

train_y = np.array(train_y)
train_x = train_x.reshape(train_x.shape[0],3072)
classifier = LDA()
classifier.fit(train_x,train_y)
LDA(n_components=None, priors=None, shrinkage=None, solver='svd',store_covariance=False, tol=0.0001)
train_x = train_x.reshape(train_x.shape[0],3,32,32)
test_x = test_batch[b'data']
test_y = test_batch[b"labels"]

dictionary = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0} 
labels = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0} 
correct_predicted = 0 
for i in range(len(test_y)):
    labels[test_y[i]] += 1
    if (classifier.predict(test_x[i].reshape(-1,3072))[0] == test_y[i]):
        correct_predicted = correct_predicted + 1
        dictionary[test_y[i]] += 1
print("Accuracy = " , (correct_predicted/len(test_y))*100 , "%")     
print("class wise accuracy = \n")
for i in range(len(dictionary)):
    print("class ",i," accuracy = ",(dictionary[i]/labels[i])*100," %")


test_x = test_x.reshape(test_x.shape[0],3,32,32)
test_y = np.array(test_y)

arr = [0,0,0,0,0,0,0,0,0,0]
i = 0
while (i < 100):
    if arr[test_y[i]] == 5:
        i = i + 1
        continue
    arr[test_y[i]] = arr[test_y[i]] + 1
    plt.imshow(np.transpose(train_x[i], (1, 2, 0))) 
    plt.show()
    i = i + 1