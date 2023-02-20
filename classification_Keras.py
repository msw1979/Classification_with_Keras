# Author Dr. M. Alwarawrah
import math, os, time, scipy
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from random import seed
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import (losses, metrics) 
from keras.utils import to_categorical
from keras.models import load_model
# import the data
from keras.datasets import mnist

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# start recording time
t_initial = time.time()

def classifier_model(num_pixels,num_classes):
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
       
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def data_classifier(num_pixels, x_train,y_train,x_test,y_test):
    # flatten training & test images
    x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32') 

    # normalize inputs from 0-255 to 0-1
    x_train = x_train / 255
    x_test = x_test / 255

    # one hot encode outputs
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    num_classes = y_test.shape[1]

    return x_train,y_train,x_test,y_test, num_classes

def plot_outpput(predictions, X_test, Y_test, name):
    plt.clf()
    N = 10#int(np.sqrt(N_classified))+1
    fig=plt.figure(figsize=(N, N))
    rows = N
    cols = N
    counter = 1
    for i in range(0,len(Y_test)):
        y_actual = np.argmax(predictions[i])
        if y_actual == Y_test[i]:
            plt.subplot(rows, cols, counter)
            plt.imshow(X_test[i])
            plt.title('{}_{}'.format(Y_test[i], y_actual))
            counter += 1
            if counter == N*N:# N_classified:
                break
    fig.tight_layout()        
    plt.savefig('test_classified_%s.png'%(name))

    plt.clf()
    N = 10#int(np.sqrt(N_misclassified))+1
    fig=plt.figure(figsize=(N, N))
    rows = N
    cols = N
    counter = 1
    for i in range(0,len(Y_test)):
        if y_actual != Y_test[i]:
            plt.subplot(rows, cols, counter)
            plt.imshow(X_test[i])
            plt.title('{}_{}'.format(Y_test[i], y_actual))
            counter += 1
            if counter == N*N:#N_misclassified:
                break
    fig.tight_layout()        
    plt.savefig('test_miclassified_%s.png'%(name))

#plot Loss and Accuracy vs epoch
def plot_loss_accuracy(train_loss, val_loss, train_acc, val_acc, name):
    plt.clf()
    fig,ax = plt.subplots()
    ax.plot(train_loss, color='k', label = 'Training Loss')
    ax.plot(val_loss, color='r', label = 'Validation Loss')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=16)
    ax2 = ax.twinx()
    ax2.plot(train_acc, color='b', label = 'Training Accuracy')
    ax2.plot(val_acc, color='g', label = 'Validation Accuracy')
    ax2.set_ylabel('Accuracy', fontsize=16)
    fig.legend(loc ="center")
    fig.tight_layout()
    plt.savefig('loss_accuracy_epoch_%s.png'%(name))

# read the data
#The MNIST database contains 60,000 training images and 10,000 testing images of digits written by high school students 
#and employees of the United States Census Bureau.
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#create a file to write the outputs
output_file = open('output.txt','w')
#print
print('x train shape: {} and y train shape: {}'.format(X_train.shape, Y_train.shape), file=output_file)
print('x test shape: {} and y test shape: {}'.format(X_test.shape, Y_test.shape), file=output_file)

#make directory for train
if os.path.exists('train') == False:
   os.mkdir('train')
#print X train with Y train in the plt title
for i in range(0,10):
    plt.clf()
    plt.imshow(X_train[i])
    plt.savefig('train/train_%s'%(Y_train[i]))

#make directory for test
if os.path.exists('test') == False:
   os.mkdir('test')
#print X test with Y test in the plt title
for i in range(0,10):
    plt.clf()
    plt.imshow(X_test[i])
    plt.savefig('test/test_%s'%(Y_train[i]))

# flatten images into one-dimensional vector
# find size of one-dimensional vector
num_pixels = X_train.shape[1] * X_train.shape[2] 
print('number of pixkes: {}'.format(num_pixels), file=output_file)

#classifier
print("Classifier Model", file=output_file)
x_train, y_train, x_test, y_test, num_classes = data_classifier(num_pixels, X_train,Y_train,X_test,Y_test)

#number of classes for test output
print('number of classes for test output: {}'.format(num_classes), file=output_file)

# build the model
classifier = classifier_model(num_pixels,num_classes)
# fit the model
epochs = 20
results = classifier.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, verbose=2)
#results    
train_loss = results.history['loss']
train_acc = results.history['accuracy']
val_loss = results.history['val_loss']
val_acc = results.history['val_accuracy']

# evaluate the model and find prediction
scores = classifier.evaluate(x_test, y_test, verbose=0)
predictions = classifier.predict(x_test)
print("Test loss:", scores[0], file=output_file)
print("Test loss:", scores[0])
print('Accuracy: %5.2f'%(scores[1]*100),'%',' & Error: %5.2f'%((1 - scores[1])*100), '%')
print('Accuracy: %5.2f'%(scores[1]*100),'%',' & Error: %5.2f'%((1 - scores[1])*100), '%', file=output_file)

#number of classified and misclassified
N_classified = int(len(Y_test)*scores[1])
N_misclassified = int(len(Y_test) - N_classified) 
#print X test with Y test (original before normalization) and prediction in the plt title
print('Number of classified: {} & Number of Misclassified: {}'.format(N_classified, N_misclassified))
print('Number of classified: {} & Number of Misclassified: {}'.format(N_classified, N_misclassified), file=output_file)

plot_outpput(predictions, X_test, Y_test, 'classifier')
plot_loss_accuracy(train_loss, val_loss, train_acc, val_acc, 'classifier')

#save and load
#model.save('classification_model_keras.h5')
#pretrained_model = load_model('classification_model.h5')

output_file.close()

#End recording time
t_final = time.time()

t_elapsed = t_final - t_initial
hour = int(t_elapsed/(60.0*60.0))
minute = int(t_elapsed%(60.0*60.0)/(60.0))
second = t_elapsed%(60.0)
print("%d h: %d min: %f s"%(hour,minute,second))