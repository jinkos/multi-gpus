import numpy as np

from keras.layers import Conv2D,Input,MaxPooling2D,Flatten,Dense
from keras.datasets import mnist
from keras import models

def get_MNIST_train_data(do_print=False):
  # get the raw test data
  (x_train, y_train), (_, _) = mnist.load_data()

  # give the images a single greyscale channel
  x_train = np.expand_dims(x_train, axis=-1)
  if do_print:
    print("x_train.shape",x_train.shape)
  
  # 1hot encode the labels
  y_train_1hot = np.zeros((60000,10))
  y_train_1hot[np.arange(60000),y_train] = 1.0
  if do_print:
    print("y_train_1hot.shape",y_train_1hot.shape)

  return (x_train, y_train_1hot)

def get_model(do_print=False):
  
  inp = Input(shape=(28,28,1))
  
  x = Conv2D(32,kernel_size=3,padding='same',strides=1,activation='relu')(inp)
  x = Conv2D(64,kernel_size=3,padding='same',strides=1,activation='relu')(x)
  x = Conv2D(128,kernel_size=3,padding='same',strides=1,activation='relu')(x)
  x = Conv2D(256,kernel_size=3,padding='same',strides=1,activation='relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = Conv2D(512,kernel_size=3,padding='same',strides=1,activation='relu')(x)
  x = Conv2D(512,kernel_size=3,padding='same',strides=1,activation='relu')(x)
  x = Conv2D(512,kernel_size=3,padding='same',strides=1,activation='relu')(x)
  x = Conv2D(512,kernel_size=3,padding='same',strides=1,activation='relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = Flatten()(x)
  x = Dense(2048, activation='relu')(x)
  x = Dense(2048, activation='relu')(x)
  x = Dense(2048, activation='relu')(x)
  x = Dense(2048, activation='relu')(x)
  x = Dense(2048, activation='relu')(x)
  x = Dense(1024, activation='relu')(x)
  x = Dense(10, activation='softmax')(x)

  model = models.Model(inputs=[inp], outputs=[x])
  
  if do_print:
    model.summary()
  
  return model
