import gzip
import numpy as np
import ssl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy.linalg import eig
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score

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
    
def principal_Component(trainX,testX,test_y,train_y,n):
    pca = PCA(n_components = n)
    nsamples, nx, ny = trainX.shape
    trainX = trainX.reshape((nsamples,nx*ny))
    nsamples, nx, ny = testX.shape
    testX = testX.reshape((nsamples,nx*ny))
    trainX = pca.fit_transform(trainX)
    testX = pca.transform(testX)
    print("Dimension of training data after pca with ",n," components = ",trainX.shape)
    print("Dimension of test data after pca with ",n," components = ",testX.shape)
    classifier = LDA()
    classifier.fit(trainX,train_y)
    LDA(n_components=None, priors=None, shrinkage=None, solver='svd',store_covariance=False, tol=0.0001)
    predicted_y = classifier.predict(testX)
    accuracy = accuracy_score(test_y,predicted_y)*100 
    return [trainX , testX]

ssl._create_default_https_context = ssl._create_unverified_context    
trainX = images_file_read("C:\pytorch_datasets\mnist\mnist\\train-images-idx3-ubyte.gz")
print("Dimension of training data",trainX.shape)
train_y = labels_file_read("C:\pytorch_datasets\mnist\mnist\\train-labels-idx1-ubyte.gz")
testX = images_file_read("C:\pytorch_datasets\mnist\mnist\\t10k-images-idx3-ubyte.gz")
test_y = labels_file_read("C:\pytorch_datasets\mnist\mnist\\t10k-labels-idx1-ubyte.gz")
labels = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0} 
dictionary = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0} 
lst = principal_Component(trainX,testX,test_y,train_y,15)
train_x = lst[0]
test_x = lst[1]

St = np.cov(np.transpose(train_x)) * (train_x.shape[0] - 1)
x_train = [[],[],[],[],[],[],[],[],[],[]]  
Sw = 0 
for i in range(train_x.shape[0]):
    x_train[train_y[i]].append(train_x[i])   
for j in range(len(x_train)):
    x_train[j] = np.array(x_train[j])
    if i == 0:    
        Sw = np.cov(np.transpose(x_train[j])) * (x_train[j].shape[0] -1)
    else:
        Sw = Sw + np.cov(np.transpose(x_train[j])) * (x_train[j].shape[0] - 1)
Sb = St - Sw                 

dimension = 1
A = np.dot(np.linalg.inv(Sw),Sb)
values, vectors = eig(A)
values = values.real
vectors = vectors.real
eiglist = [(values[i],vectors[:,i]) for i in range(len(values))]
eiglist = sorted(eiglist, key = lambda x: x[0], reverse = True)
W = np.array([eiglist[i][1] for i in range(dimension)])  
Y = np.dot(W, np.transpose(train_x))

classifier = LDA()
classifier.fit(np.transpose(Y) , train_y) 
LDA(n_components=None, priors=None, shrinkage=None, solver='svd',store_covariance=False, tol=0.0001)

test = np.array(W.dot(np.array(test_x).transpose())).transpose()
predicted_y = classifier.predict(test)
accuracy = accuracy_score(test_y,predicted_y)*100 
print("Accuracy with pca followed by fda = " , accuracy, "%")
for i in range(len(test_y)):
    labels[test_y[i]] += 1
    if classifier.predict(test[i].reshape(-1,dimension))[0] == test_y[i]:
        dictionary[test_y[i]] += 1
for i in range(len(labels)):
    print("class ",i," accuracy = ",(dictionary[i]/labels[i])*100," %")        
