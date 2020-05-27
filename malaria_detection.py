!pip install tensorflow-gpu==2.0.0-rc0

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, ZeroPadding2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

!git clone https://github.com/Shobhit2526/Malaria-Detection.git

img_width = 64
img_heigth = 64

datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)

train_data_generator = datagen.flow_from_directory(directory='/content/Malaria-Detection/Dataset',
                                                   target_size = (64,64),
                                                   class_mode = 'binary',
                                                   batch_size = 16,
                                                   subset = 'training'
                                                   )

validation_data_generator = datagen.flow_from_directory(directory='/content/Malaria-Detection/Dataset',
                                                   target_size = (64,64),
                                                   class_mode = 'binary',
                                                   batch_size = 16,
                                                   subset = 'validation'
                                                   )

train_data_generator.labels

model = Sequential()

model.add(Conv2D(16,(3,3),input_shape = (img_width,img_heigth,3), activation = 'relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(32,(3,3), activation = 'relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(optimizers='adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

history = model.fit_generator(generator = train_data_generator,
                              steps_per_epoch = len(train_data_generator),
                              epochs = 5,
                              validation_data = validation_data_generator,
                              validation_steps = len(validation_data_generator))

history.history

def plot_learningCurve(history,epoch):
  #plot training & validation accuracy values
   epoch_range = range(1,epoch+1)
   plt.plot(epoch_range,history.history['accuracy'])
   plt.plot(epoch_range,history.history['val_accuracy'])
   plt.title('Model_Accuracy')
   plt.ylabel('Accuracy')
   plt.xlabel('Epoch')
   plt.legend(['Train','Val'],loc = 'upper left')
   plt.show()
#plot training & validtaion loss values
   plt.plot(epoch_range,history.history['loss'])
   plt.plot(epoch_range,history.history['val_loss'])
   plt.title('Model_loss')
   plt.ylabel('Loss')
   plt.xlabel('Epoch')
   plt.legend(['Train','Val'],loc = 'upper left')
   plt.show()

plot_learningCurve(history,5)