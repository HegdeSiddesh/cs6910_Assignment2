"""
#CS6910 Assignment 2 Part B: 
The goal of this part is as follows:        
     
- Finetune a pre-trained model just as you would do in many real world applications

### Import required packages
"""

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import math
from keras.layers import Conv2D , MaxPool2D , Flatten , Dropout, Dense, Activation, BatchNormalization, GlobalAveragePooling2D

import random
np.random.seed(137) # To ensure that the random number generated are the same for every iteration
import warnings
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
import os
from keras.datasets import fashion_mnist
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten, InputLayer
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation
from keras.callbacks import EarlyStopping
tf.random.set_seed(137)


filePath = 'C:/Users/hegde/Desktop/Test_codes/nature_12k/inaturalist_12K'

"""###Create functions for creating models using pretrained models"""

class ConfigValues():
    """
    Class to hold the parameters needed to create a model using the pretrained models.

    Attributes:
        layers_unfreeze:Number of layers from end of pretrained model to be used while training (default 20)
        model:The pretrained model to be used (default "InceptionResNetV2")
        dense_layers:Neurons in the dense layer(default [128])
        epochs:Number of epochs to tun the model (default 10)
        batch_size: Batch size (default 256)
        augment_data:Augment data or not (default 'no')
        dropout: Dropout rate (default 0.2)
    """
    def __init__(self, augment_data='no', batch_size=256, dense_layers=[128], dropout=0.2, epochs=10, layers_unfreeze=20, model="InceptionResNetV2"):
      self.augment_data = augment_data
      self.batch_size = batch_size
      self.dense_layers = dense_layers
      self.dropout = dropout
      self.epochs = epochs
      self.layers_unfreeze = layers_unfreeze
      self.model = model

def generateData(augment_data_str='no', batch_size = 64):
  """
  Function to generate the train and validation dataset based on user parameters

  Attributes:
      augment_data_str: Augment data or not (default 'no')
      batch_size: Batch size (default 64)
  """

  if augment_data_str =='no':
    print("Not augmenting data")
    augment_data = False
  else:
    print("Augmenting data")
    augment_data = True

  #Generate the train and validation data
  #Validation split is 10% of train data
  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,validation_split=0.1)
  IMG_SIZE = (256,256)

  if augment_data==True:  
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,rotation_range=90, shear_range=0.2, height_shift_range=0.2,
                                          width_shift_range=0.2,
                                          horizontal_flip=True,
                                          zoom_range=0.2,validation_split=0.1)

  else:
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,validation_split=0.1)


  train_gen = train_datagen.flow_from_directory(
          filePath+'/train',
          target_size=IMG_SIZE,
          subset = 'training',
              batch_size=batch_size,
              class_mode='categorical',
              shuffle = True,
          seed = 137)
  print('TRAINING')
  print('Number of samples', train_gen.samples)
  print('Names of classes', train_gen.class_indices)
  print('Number of classes', len(train_gen.class_indices))
  print('Number of samples per class', int(train_gen.samples / len(train_gen.class_indices) ))

  validation_gen = train_datagen.flow_from_directory(
          filePath+'/train',
          target_size=IMG_SIZE,
              subset = 'validation',
              batch_size=batch_size,
              class_mode='categorical',
              shuffle = False,
          seed = 137)
  print('VALIDATION')
  print('Number of samples', validation_gen.samples)
  print('Names of classes', validation_gen.class_indices)
  print('Number of classes', len(validation_gen.class_indices))
  print('Number of samples per class', int(validation_gen.samples / len(validation_gen.class_indices) ))



  train_generator = tf.data.Dataset.from_generator(
    lambda: train_gen,
    output_types = (tf.float32, tf.float32)
    ,output_shapes = ([None, 256, 256, 3], [None, 10]),
  )
  train_generator = train_generator.repeat()
  train_generator = train_generator.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  validation_generator = tf.data.Dataset.from_generator(
      lambda: validation_gen,
      output_types = (tf.float32, tf.float32)
      ,output_shapes = ([None, 256, 256, 3], [None, 10]),
  )
  validation_generator = validation_generator.repeat()
  validation_generator = validation_generator.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return train_gen, train_generator, validation_gen, validation_generator

def executePretrainedModel(parameters, log = False):
  """
  Function to create and execute the model based on user parameters

  Attributes:
      parameters: Object holding the parameter values
      log : Log metrics onto wandb or not (default False)
  """
  IMG_SIZE = (256,256)
  IMG_SHAPE = IMG_SIZE + (3,)

  train_gen, train_generator, validation_gen, validation_generator = generateData(parameters.augment_data, parameters.batch_size )

  #######################################################Set the model (pretrained + custom)##########################################################################

  input_ = tf.keras.Input(shape = IMG_SHAPE)

  # add a pretrained model without the top dense layer
  if parameters.model == 'ResNet50':
    pretrained_model = tf.keras.applications.ResNet50(include_top = False, weights='imagenet',input_tensor = input_)
  elif parameters.model == 'InceptionV3':
    pretrained_model = tf.keras.applications.InceptionV3(include_top = False, weights='imagenet',input_tensor = input_)
  elif parameters.model == 'InceptionResNetV2':
    pretrained_model = tf.keras.applications.InceptionResNetV2(include_top = False, weights='imagenet',input_tensor = input_)
  elif parameters.model == 'Xception':
    pretrained_model = tf.keras.applications.Xception(include_top = False, weights='imagenet',input_tensor = input_)
  elif parameters.model == 'MobileNetV2':
    pretrained_model = tf.keras.applications.MobileNetV2(include_top = False, weights='imagenet',input_tensor = input_)

  #freeze all layers
  for layer in pretrained_model.layers:
      layer.trainable=False 

  #Make last "x" layer trainable from pretrained model
  if parameters.layers_unfreeze is not None:   
    print("Training last ",parameters.layers_unfreeze," layers")
    for layer in pretrained_model.layers[-parameters.layers_unfreeze:]:
      layer.trainable=True

  model = tf.keras.models.Sequential()
  #Add pretrained model
  model.add(pretrained_model)

  #Add layers to flatten above outputs
  model.add(GlobalAveragePooling2D())
  #Add dropout
  model.add(Dropout(parameters.dropout)) 

  #Add the dense layers as passed by user
  for i in parameters.dense_layers:
    model.add(Dense(i, activation='relu'))#add a dense layer
    model.add(Dropout(parameters.dropout)) # For dropout
  model.add(Dense(10, activation="softmax"))#softmax layer

  model.summary()

  ###########################################################################################################################

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

  #Early stopping added to the model
  earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')


  #number of epochs for pre training 
  hist = model.fit(train_generator,
              epochs=parameters.epochs,
              batch_size=parameters.batch_size,
              steps_per_epoch = train_gen.samples//train_gen.batch_size,
              validation_data= validation_generator,
              validation_steps= validation_gen.samples//validation_gen.batch_size,
              callbacks=[earlyStopping]
              )
  
  val_acc=max(hist.history['val_accuracy'])
  train_acc=max(hist.history['accuracy'])
  val_loss=max(hist.history['val_loss'])
  train_loss=max(hist.history['loss']) 

  log_values = {'Training_loss':train_loss, 'Validation_loss':val_loss, 'Training_accuracy':train_acc, 'Validation_accuracy':val_acc}

  #Log metrics onto wandb if required
  #if log==True:
    #wandb.log(log_values)
    #wandb.log({"accuracy": val_acc})

  print(log_values)


  return model

"""##Create model for Transfer learning

The hyperparameters of the model are as follows:

- augment_data
- batch_size
- dense_layers
- dropout
- epochs
- layers_unfreeze
- model
"""

augment = input("Perform data augmentation(enter yes or no):")

if augment =="yes":
	augment_data= True
else: 
	augment_data= False

batch_size = int(input("Enter batch size:"))

import ast

dense_layers = ast.literal_eval(input("Enter dense layer sizes as a list. eg:[64,128] :"))

dropout = float(input("Enter dropout rate (between 0 and 1):"))

epochs= int(input("Enter number of epochs:"))

layers_unfreeze= int(input("Enter number of layers to fine tune from the pretrained model:"))

model_num = int(input("Enter number based on desired pretrained model (1:ResNet50, 2:InceptionV3, 3:InceptionResNetV2, 4:Xception, 5:MobileNetV2):"))

if model_num>5 or model_num<1 :
	model_name= "InceptionResNetV2"
else: 
  if model_num==1:
    model_name = "ResNet50"
  if model_num==2:
    model_name = "InceptionV3"
  if model_num==3:
    model_name = "InceptionResNetV2"
  if model_num==4:
    model_name = "Xception"
  if model_num==5:
    model_name = "MobileNetV2" 


parameters = ConfigValues(augment_data=augment_data, batch_size=batch_size, dense_layers=dense_layers, dropout=dropout, epochs=epochs, layers_unfreeze=layers_unfreeze, model=model_name)

model = executePretrainedModel(parameters, log = False)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
          filePath+'/val',
          target_size=(256,256),
              batch_size=parameters.batch_size,
              class_mode='categorical',
              shuffle = False,
          seed = 137)
test_generator = tf.data.Dataset.from_generator(
      lambda: test_gen,
      output_types = (tf.float32, tf.float32)
      ,output_shapes = ([None, 256, 256, 3], [None, 10]),
  )
test_generator = test_generator.repeat()
test_generator = test_generator.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_loss, test_acc = model.evaluate(test_generator, steps=test_gen.samples//test_gen.batch_size, verbose=2)
print('Test accuracy :', test_acc)

"""
##Save model
"""

MODEL_PATH = filePath+'/best_model_pretrained.h5'

#Save Model
model.save(MODEL_PATH)

# Load Model
#model = load_model(MODEL_PATH)