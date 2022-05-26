import pandas as pd
import numpy as np
import ssl
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import keras
import tensorflow as tf

def classWiseAcc(test_x, test_y):
    saved_model = keras.models.load_model('best_model.h5')
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
        
    
ssl._create_default_https_context = ssl._create_unverified_context
train_set = pd.read_csv(r'C:\pytorch_datasets\Fmnist_dataset\\fashion-mnist_train.csv')
test_set = pd.read_csv(r'C:\pytorch_datasets\Fmnist_dataset\\fashion-mnist_test.csv')
label = {0:"T-shirt/top",1 :"Trouser",2 :"Pullover",3 :"Dress",4 :"Coat",5 :"Sandal",6 :"Shirt",7 :"Sneaker",8 :"Bag",9 :"Ankle boot"}
train_x = np.array(train_set.iloc[:,1:]).reshape(train_set.shape[0],28,28)
train_y = np.array(train_set.iloc[:,0])
test_x = np.array(test_set.iloc[:,1:]).reshape(test_set.shape[0],28,28) 
test_y = np.array(test_set.iloc[:,0])
val_x = test_x[0:500,:,:].reshape(-1,784) / 255 - 0.5
val_y = test_y[0:500] 
test_images = test_x.reshape(-1,784) / 255 - 0.5

model = keras.Sequential()
model.add(keras.layers.Dense(64, input_shape=(784,), activation="relu"))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
sgd = tf.keras.optimizers.SGD(learning_rate = 0.01) 
weights = np.array(model.get_weights())
for _ in range(0,6,2):
    for i in range(weights[_].shape[0]):
        for j in range(weights[_].shape[1]):
            weights[_][i][j] = random.uniform(-0.5,0.5)        
model.set_weights(weights)
model.compile(
  optimizer = sgd,
  loss = 'categorical_crossentropy',
  metrics = ['accuracy'], 
)

es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5) 
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
H = model.fit(train_x.reshape(-1,784)/255 - 0.5,to_categorical(train_y),validation_data=(val_x.reshape(-1,784), to_categorical(val_y)),  epochs=10,batch_size=128,shuffle = True,callbacks=[es,mc],)
print(model.summary()) 
num_Epochs = [0,1,2,3,4,5,6,7,8,9]
plt.bar(num_Epochs,list(H.history['loss']))
plt.xlabel("Epochs")
plt.ylabel("Training Loss") 
plt.show()

saved_model = keras.models.load_model('best_model.h5')
print('Test accuracy:', saved_model.evaluate(test_images, to_categorical(test_y))[1])  
Accuracy = classWiseAcc(test_x, test_y)
print("Class Wise Accuracy: ")
for i in range(len(Accuracy)):
    print("Accuracy of ", i, " class = ", Accuracy[i]*0.1," %")
