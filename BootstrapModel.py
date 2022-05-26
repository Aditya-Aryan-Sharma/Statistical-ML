import gzip
import numpy as np
import ssl
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


def images_file_read(file_name):
    with gzip.open(file_name, 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        image_count = int.from_bytes(f.read(4), 'big')
        row_count = int.from_bytes(f.read(4), 'big')
        column_count = int.from_bytes(f.read(4), 'big')
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
        return images
    
def labels_file_read(file_name):
    with gzip.open(file_name, 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        label_count = int.from_bytes(f.read(4), 'big')
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels       

def classWiseAcc(labels, test_y):
    classAcc = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0} 
    total = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0} 
    for i in range (len(labels)):
        total[test_y[i]] += 1
        if (test_y[i] == labels[i]):
            classAcc[test_y[i]] += 1 
    for j in classAcc.keys():
        print("Class Accuracy for class ",j," = ",(classAcc[j]/total[j])*100)    
            
    
ssl._create_default_https_context = ssl._create_unverified_context    
train_x = images_file_read("C:\pytorch_datasets\mnist\mnistDataset\mnist\\train-images-idx3-ubyte.gz")
train_y = labels_file_read("C:\pytorch_datasets\mnist\mnistDataset\mnist\\train-labels-idx1-ubyte.gz")
test_x = images_file_read("C:\pytorch_datasets\mnist\mnistDataset\mnist\\t10k-images-idx3-ubyte.gz")
test_y = labels_file_read("C:\pytorch_datasets\mnist\mnistDataset\mnist\\t10k-labels-idx1-ubyte.gz") 

bag_size = 3
split = 15000
prediction = list()
for k in range(test_x.shape[0]):
    prediction.append(list())
  
for i in range(bag_size):
    classifier_model = DecisionTreeClassifier(max_depth = 4)
    classifier_model.fit(train_x[split*i:split*(i+2)].reshape(-1,784),train_y[split*i:split*(i+2)]) 
    for j in range(test_x.shape[0]):
        prediction[j].append(classifier_model.predict(test_x[j].reshape(-1,784))[0]) 

labels = list()
for i in range(len(prediction)):
    dictionary = {}
    for j in range(len(prediction[i])):
        if (prediction[i][j] in dictionary.keys()):
            dictionary[prediction[i][j]] += 1
        else:
            dictionary[prediction[i][j]] = 1   
    max , index = 0 , 0    
    for k in dictionary.keys():
        if (max < dictionary[k]):
            max = dictionary[k]
            index = k 
    labels.append(index)
print("Testing Accuracy: ",accuracy_score(np.array(labels),test_y)*100) 
print("Class Wise Testing Accuracy:\n")
classWiseAcc(labels,test_y) 
