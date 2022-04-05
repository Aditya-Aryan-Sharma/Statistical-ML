import gzip
import numpy as np
import ssl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
    
def principal_Component(train_x,test_x,test_y,train_y,n):
    pca = PCA(n_components = n)
    nsamples, nx, ny = train_x.shape
    train_x = train_x.reshape((nsamples,nx*ny))
    nsamples, nx, ny = test_x.shape
    test_x = test_x.reshape((nsamples,nx*ny))
    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)
    print("Dimension of training data after pca with ",n," components = ",train_x.shape)
    classifier = LDA()
    classifier.fit(train_x,train_y)
    LDA(n_components=None, priors=None, shrinkage=None, solver='svd',store_covariance=False, tol=0.0001)
    predicted_y = classifier.predict(test_x)
    accuracy = accuracy_score(test_y,predicted_y)*100 
    print("Accuracy = " , accuracy, "%")
    return accuracy
    
ssl._create_default_https_context = ssl._create_unverified_context    
train_x = images_file_read("C:\pytorch_datasets\mnist\mnist\\train-images-idx3-ubyte.gz")
print("Dimension of training data",train_x.shape)
train_y = labels_file_read("C:\pytorch_datasets\mnist\mnist\\train-labels-idx1-ubyte.gz")
test_x = images_file_read("C:\pytorch_datasets\mnist\mnist\\t10k-images-idx3-ubyte.gz")
test_y = labels_file_read("C:\pytorch_datasets\mnist\mnist\\t10k-labels-idx1-ubyte.gz")

dictionary = {3:0,8:0,15:0}
dictionary[15] = principal_Component(train_x,test_x,test_y,train_y,15)
dictionary[8] = principal_Component(train_x,test_x,test_y,train_y,8)
dictionary[3] = principal_Component(train_x,test_x,test_y,train_y,3)
n_values = list(dictionary.keys())
n_accuracy = list(dictionary.values())
plt.bar(n_values, n_accuracy)
plt.xlabel("number of components")
plt.ylabel("Accuracy")
plt.show()