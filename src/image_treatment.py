
import os
from os.path import isfile, join
import pandas as pd
import numpy as np
from skimage.io import imread
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.io import imread
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.applications.vgg16 import VGG16

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Activation

from keras.layers.core import Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras import backend as K


class DataTransformation():
    
    def augment_data(path:str,datagen, target: str, rang: int, augm: int)->None:
        '''
            This function receives a data generator and target and returns
            a number of copies of the same image with different transformations,
            to increment the number of images for train our model

            Parameters:
                path: str --> path to find the images and save the new images
                datagen: Object generator with the features to transform the image
                target: str -->  Name of target images class
                rang: int --> Number of the images you want aumentate
                augm: int --> Number of aumentated transformations per image

            Returns:
                A message of work is done
        '''

        # first create a list of the files on the target folder
        target_lst = [f'{path}{target}/' + f for f in os.listdir(f'{path}{target}/') if isfile(join(f'{path}{target}/', f))]
        # path for save the augmentated data
        save_here = f'{path}{target}/aug'
        try:
            os.stat(save_here) # if folder exits save on it
        except:
            os.mkdir(save_here) # if not, create it and save on it
        for i in tqdm(range(rang)):
            # transformation the image
            image = np.expand_dims(imread(target_lst[i]), axis=0) 
            datagen.fit(image)
            # makes the augmetation
            for x, val in zip(datagen.flow(image,                     # image we choose
                                      save_to_dir=save_here,          # the folder on we save the new image 
                                      save_prefix='aug',               
                                      save_format='png'), range(augm)): # number of augmented images we want
                pass
        return 'Augmentation Finished'

    def transform_data(path: str, classes: list, size:int, neural_network: bool = False, test: bool = False )->(np.array):
            
        '''
            Function that takes the images, resize it and transform to a numpy array, then shuffle it and split the
            data in train and test to feed the models.
                
            Parameters:
                path: The path that stores the images folders
                classes: List with the different classes of images we want to be classified for test you must give an 
                            empty list
                size: New size for the images
                 multilevel array, with one colum for each class, default False
                test: bool if you want to transform the images for test you must give a path for the images test folder
                        and the classes attribute as an empty list
            Return:
                
                X_train, X_test, y_train and y_test
                    
                if neural_network = True:
                    X_train, X_test, Y_train, Y_test
        '''
            
            
        X = []
        Y = []
            
        if test:
            try:
                if len(classes) == 0:
                    for file in tqdm(os.listdir(f'{path}')):
                        image = imread(f'{path}'+file)
                        smallimage = cv2.resize(image,(size,size))
                        X.append(smallimage)

                    X = np.array(X)

                    X = X/255.0 # normalize X

                    return X
                else:
                    
                    raise ValueError(f'''For test you must past the path of test images and an empty list for class attribute 
                                    expected length 0, length received {len(classes)}''')        
            except:
                raise ValueError(f'''For test you must past the path of test images and an empty list for class attribute 
                                    expected length 0, length received {len(classes)}''')
                
                
        else:
            
            for i,c in enumerate(classes):
                for file in tqdm(os.listdir(f'{path}{c}/aug/')):
                    image = imread(f'{path}{c}/aug/'+file)
                    smallimage = cv2.resize(image,(size,size))
                    X.append(smallimage)
                    Y.append(i)

            X,Y = np.array(X), np.array(Y) # transform X & Y, to a numpy array

            X = X/255.0 # normalize X

            X,Y = shuffle(X,Y, random_state=42) # shuffle the data

            X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = .2, random_state=42, stratify=Y)

            if neural_network == True: #if we want to train a neural network need to transform or target columns to categories
                
                Y_train = np_utils.to_categorical(y_train,len(classes))
                
                Y_test = np_utils.to_categorical(y_test,len(classes))

                return X_train, X_test, Y_train, Y_test

            else:

                return X_train, X_test, y_train, y_test
            

class TrainModel():
    
    def __init__(self, size=32, kernel_size = 4, pool_size=2, model = Sequential()):
        
        self.size = size
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.model = model
    
    def fit_model(self, X_train, Y_train, X_test, Y_test, class_length, batch_size = 64, epochs=5, verbose=1):
        
        def recall_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        def f1_m(y_true, y_pred):
            precision = precision_m(y_true, y_pred)
            recall = recall_m(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))
        
        
        self.model.add(Convolution2D(self.size, # Number convolution channels to generate
                        (self.kernel_size, self.kernel_size), # Size of convolution kernels
                        padding='valid', # Strategy to deal with borders
                        input_shape=(self.size, self.size, class_length))) # Size = image rows x image columns x channels
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(self.size, # Number convolution channels to generate
                                (self.kernel_size, self.kernel_size), # Size of convolution kernels
                                padding='valid', # Strategy to deal with borders
                                input_shape=(self.size, self.size, class_length))) # Size = image rows x image columns x channels
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(self.pool_size, self.pool_size)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(class_length))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy", f1_m, recall_m, precision_m])
        history = self.model.fit(
            X_train, # Training data
            Y_train, # Labels of training data
            batch_size= batch_size, # Batch size for the optimizer algorithm
            epochs=epochs, # Number of epochs to run the optimizer algorithm
            verbose=verbose # Level of verbosity of the log messages
        )
        score = self.model.evaluate(X_test, Y_test)
        print("Test loss", score[0])
        print("Test accuracy", score[1])

        pd.DataFrame(history.history).plot(figsize=(8,5))
        plt.grid(True)
        plt.show()
        
        return self.model
        

    def predictions(self, test, classes):
        
        preds = []
        
        for i in range(len(test)):
            pred = np.argmax(self.model.predict(np.expand_dims(test[i], axis=0)))
            
            preds.append(pred)
            
        mod = len(test)%10
        if mod != 0:
            rows_plot = len(test)//10 + 1
            cols = 10
        else:
            rows_plot = len(test)//10 
            cols = 10
            
        for i in range(len(test)):
            plt.figure(figsize=(30,20))
            plt.subplot(rows_plot,cols,i+1)
            plt.xticks([])
            plt.yticks([])
            for j,c in enumerate(classes):
                if preds[i] == j:
                    plt.xlabel(c)
                    plt.imshow(test[i])
            
        plt.show()
        
        return preds

class Train_Transfer_Learning_Model():
    
    def __init__(self, size = 32, num_targets = 3):
        
        self.size = size
        self.num_targets = num_targets
        self.model = VGG16(input_shape=(self.size, self.size, self.num_targets),
                          include_top = False,
                          weights = 'imagenet')
        
    def fit_model(self, X_train, Y_train, X_test, Y_test, batch_size = 64, epochs = 5, verbose = 1):
        
        for layer in self.model.layers:
            layer.trainable = False
            
        def recall_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        def f1_m(y_true, y_pred):
            precision = precision_m(y_true, y_pred)
            recall = recall_m(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))
        
        x = layers.Flatten()(self.model.output)

        x = layers.Dense(512, activation='relu')(x)
        
        x = layers.Dense(self.num_targets, activation='sigmoid')(x)

        self.model = tf.keras.models.Model(self.model.input, x)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['acc', f1_m, recall_m, precision_m])
        
        vgghist = self.model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs = epochs,
                            verbose = verbose)

        pd.DataFrame(vgghist.history).plot(figsize=(8,5))
        plt.grid(True)
        plt.show()
        
        return self.model
    
    def predictions(self, test, classes):
        
        preds = []
        
        for i in range(len(test)):
            
            pred = np.argmax(self.model.predict(np.expand_dims(test[i], axis=0)))
            
            preds.append(pred)
        
        mod = len(test)%10
        if mod != 0:
            rows_plot = len(test)//10 + 1
            cols = 10
        else:
            rows_plot = len(test)//10 
            cols = 10
            
        for i in range(len(test)):
            plt.figure(figsize=(30,20))
            plt.subplot(rows_plot,cols,i+1)
            plt.xticks([])
            plt.yticks([])
            for j,c in enumerate(classes):
                if preds[i] == j:
                    plt.xlabel(c)
                    plt.imshow(test[i])
            
        plt.show()
        
        return preds    