import gzip
import numpy as np
import ssl
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score
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
    
def compute(predicted):
    for i in range(len(predicted)):
        predicted[i] = round(predicted[i])
    return predicted        

ssl._create_default_https_context = ssl._create_unverified_context    
train_x = images_file_read("C:\pytorch_datasets\mnist\mnist_dataset\mnist\\train-images-idx3-ubyte.gz")
train_y = labels_file_read("C:\pytorch_datasets\mnist\mnist_dataset\mnist\\train-labels-idx1-ubyte.gz")
test_x = images_file_read("C:\pytorch_datasets\mnist\mnist_dataset\mnist\\t10k-images-idx3-ubyte.gz")
test_y = labels_file_read("C:\pytorch_datasets\mnist\mnist_dataset\mnist\\t10k-labels-idx1-ubyte.gz")

reg_model = DecisionTreeRegressor(max_depth = 2)
reg_model.fit(train_x.reshape(-1,784),train_y)
num_iter = 5
learning_rate = 0.5
testAccuracy = {1:0,2:0,3:0,4:0,5:0}
trainAccuracy = {1:0,2:0,3:0,4:0,5:0}
predicted = reg_model.predict(train_x.reshape(-1,784))
predicted = compute(predicted)
train_label = reg_model.predict(train_x.reshape(-1,784))
test_label = reg_model.predict(test_x.reshape(-1,784)) 

for i in range(num_iter):
    for j in range(len(test_y)):
        test_label[j] = test_label[j] + (test_y[j] - test_label[j]) * learning_rate
    for k in range (len(train_y)):
        train_label[k] = train_label[k] + (train_y[k] - train_label[k]) * learning_rate 
    train_accuracy = accuracy_score(compute(train_label),train_y)*100
    test_accuracy = accuracy_score(compute(test_label),test_y)*100
    testAccuracy[i+1] = test_accuracy
    trainAccuracy[i+1] = train_accuracy
    
print("Final Train Accuracy = ", trainAccuracy[5])
print("Final Test Accuracy = ", testAccuracy[5]) 

plt.xlabel('Iterations')
plt.ylabel('Test Accuracy')
n_values = list(testAccuracy.keys())
n_accuracy = list(testAccuracy.values())
plt.bar(n_values, n_accuracy)   
plt.show()

plt.xlabel('Iterations')
plt.ylabel('Train Accuracy')
n_values = list(trainAccuracy.keys())
n_accuracy = list(trainAccuracy.values())
plt.bar(n_values, n_accuracy)   
plt.show() 
