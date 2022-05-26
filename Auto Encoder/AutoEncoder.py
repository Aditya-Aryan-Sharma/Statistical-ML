import gzip
import numpy as np
import ssl
from sklearn.metrics import accuracy_score
import keras 
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint 
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf

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

def classWiseAcc(test_x, test_y):
    saved_model = keras.models.load_model('saved_model.h5')
    classAcc = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0} 
    test_x = test_x.reshape(-1,784) / 255 - 0.5
    predicted_y = saved_model.predict(test_x)
    for i in range(predicted_y.shape[0]):
        maxVal = -1
        index = -1
        for j in range(10):
            if (maxVal < predicted_y[i][j]):
                maxVal = predicted_y[i][j]
                index = j
            if (j == 9):
                if (test_y[i] == index):
                    classAcc[index] = classAcc[index] + 1    
    return classAcc 

def classSamples(test_y):
    total = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0} 
    for i in range(test_y.shape[0]):
        total[test_y[i]] = total[test_y[i]] + 1 
    return total

ssl._create_default_https_context = ssl._create_unverified_context    
train_x = images_file_read("C:\pytorch_datasets\mnist\mnist_data\mnist\\train-images-idx3-ubyte.gz")
train_y = labels_file_read("C:\pytorch_datasets\mnist\mnist_data\mnist\\train-labels-idx1-ubyte.gz")
test_x = images_file_read("C:\pytorch_datasets\mnist\mnist_data\mnist\\t10k-images-idx3-ubyte.gz")
test_y = labels_file_read("C:\pytorch_datasets\mnist\mnist_data\mnist\\t10k-labels-idx1-ubyte.gz")
val_x = test_x[0:500,:,:].reshape(-1,784) / 255 - 0.5
val_y = test_y[0:500] 
test_images = test_x.reshape(-1,784) / 255 - 0.5 

model = keras.Sequential()
model.add(keras.layers.Dense(512, input_shape=(784,), activation="relu"))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dense(784, activation="relu")) 
adam = tf.keras.optimizers.Adam(learning_rate = 0.6, beta_1=0.9, beta_2=0.9, epsilon=1e-1) 
model.compile(
  optimizer = adam,
  loss = 'mse',
  metrics = ['accuracy'], 
)
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5) 
mc = ModelCheckpoint('saved_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
H = model.fit(train_x.reshape(-1,784)/255 ,train_x.reshape(-1,784)/255 ,validation_data=(val_x.reshape(-1,784), val_x.reshape(-1,784)),  epochs=5,batch_size=64,shuffle = True,callbacks=[es],) 
print(model.summary()) 

epochs = [1,2,3,4,5]
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.bar(epochs,list(H.history["loss"])) 
plt.show()

for i in range(3):
    model.pop()
print("Training MNIST classifier model:")
model.add(keras.layers.Dense(32, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
adam_opt = tf.keras.optimizers.Adam(learning_rate = 0.01, decay = 1e-6)
model.compile(
    optimizer = adam_opt,
    loss = "categorical_crossentropy",
    metrics = ['accuracy'], 
)
Hist = model.fit(train_x.reshape(-1,784)/255 - 0.5,to_categorical(train_y),validation_data=(val_x.reshape(-1,784), to_categorical(val_y)),  epochs=5,batch_size=128,shuffle = True,callbacks=[es,mc],)
num_Epochs = [1,2,3,4,5]
plt.bar(num_Epochs,list(Hist.history['loss']))
plt.xlabel("Epochs")
plt.ylabel("Training Loss")  
plt.show()

saved_model = keras.models.load_model('saved_model.h5')
print('Test accuracy:', saved_model.evaluate(test_images, to_categorical(test_y))[1])  
Accuracy = classWiseAcc(test_x, test_y)
total = classSamples(test_y)
print("Class Wise Accuracy: ")
for i in range(len(Accuracy)):
    print("Accuracy of ", i, " class = ", (Accuracy[i]/total[i])*100," %") 
