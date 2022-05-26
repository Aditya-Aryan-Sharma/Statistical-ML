import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
import sys
import pickle as cPickle
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.layers import Flatten, Reshape
from keras.layers import Dense, Activation
from keras.models import Model
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
import tensorflow as tf

f = gzip.open('mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    data = cPickle.load(f)
else:
    data = cPickle.load(f, encoding='bytes')
f.close()
(train_images, train_labels), (test_images, test_labels) = data

train_images = train_images / np.max(train_images)
test_images = test_images / np.max(test_images)
train_images = train_images.reshape(-1, 28,28, 1)
test_images = test_images.reshape(-1, 28,28, 1)
train_images.shape, test_images.shape

from sklearn.model_selection import train_test_split
train_X,valid_X,train_ground,valid_ground = train_test_split(train_images,train_images, test_size=0.8, random_state=13)
batch_size = 128
epochs = 1
inChannel = 1
x, y = 28, 28
input_img = Input(shape = (x, y, inChannel))

model = Sequential()
model.add(Conv2D(4, (3,3), input_shape=(input_img[0].shape))) 
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
conv1 = Conv2D(4, (3, 3), activation='relu', padding='same')(input_img)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv3')(pool2)
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3) 
up1 = UpSampling2D((2,2))(conv3) 
conv5 = Conv2D(8, (3, 3), activation='relu', padding='same')(up1) 
up2 = UpSampling2D((2,2))(conv5) 
bnup2 = BatchNormalization()(up2)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(bnup2)
autoencoder = Model(input_img, decoded)
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
autoencoder.summary()
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))

loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

pred = autoencoder.predict(test_images)
pred.shape
plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(test_images[i, ..., 0], cmap='gray')
    curr_lbl = test_labels[i]
plt.show()    

plt.figure(figsize=(20, 4))
print("Reconstruction of Test Images")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(pred[i, ..., 0], cmap='gray')  
plt.show()
for layer in autoencoder.layers[:-5]:
    print(layer.name)
    layer.trainable = False
flat = Flatten()(conv3)
output = Dense(10,activation='softmax')(flat)
encoder = Model(input_img,output)
encoder.compile(optimizer='adam',loss='categorical_crossentropy',
                    metrics=['accuracy'])
encoder.summary()
encoder.fit(train_images,to_categorical(train_labels),epochs=1,batch_size=32)
score = encoder.evaluate(test_images, to_categorical(test_labels), verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
