# written by Mohammad Reza Kakoee 

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from all_functions import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Activation,Dropout
from keras.layers.convolutional import Convolution2D,MaxPooling2D,Cropping2D
from keras.layers import Conv2D
import pickle
from all_functions import *



## debug
debug_train=0
debug_save_model =1
debug_custom_train_extract=1
##

            

#Training

images_notped = glob.glob('./Train_dataset/nocar_bike_ped/*.png')
images_pedcar = glob.glob('./Train_dataset/ped_bike/*.png')

print(len(images_notped))


cars = np.array([])
notcars = np.array([])
for image in images_notped:
    notcars=np.append(notcars,image)
    
for image in images_pedcar:
    cars=np.append(cars,image)

    
np.random.shuffle(cars) 
np.random.shuffle(notcars)

notcars=notcars[0:(int)(len(cars)/2)]

cars = np.array([])
notcars = np.array([])
 
#custom_train_car
images_cars_custom = glob.glob('./Train_dataset/custom_ped/*.png')
for image in images_cars_custom:
    cars=np.append(cars,image)

#custom_train_nocar extraction
if(debug_custom_train_extract==1):
    images_notcars_custom = glob.glob('./Train_dataset/nopedbike*')
    cnt=0
    for imagename in images_notcars_custom:
        not_cars_c=crop_images(imagename,64)
        cnt+=1
        for index,image in enumerate(not_cars_c):
            filename="./Train_dataset/custom_noped/custom_nocar"+str(cnt)+"_"+str(index)+".png"
            cv2.imwrite(filename,image)

#custom_train_notcar
images_notcars_custom = glob.glob('./Train_dataset/custom_noped/custom_nocar*')
for image in images_notcars_custom:
    notcars=np.append(notcars,image)

 

model_filename = 'finalized_model_NN_rgb.sav'
BATCH_SIZE=16

# Create an array stack of feature vectors
X = np.hstack((cars, notcars))

# Define the labels vector
y = np.hstack((np.ones(len(cars),dtype=int), np.zeros(len(notcars),dtype=int)))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)


if(debug_train==1):
    
    
    # compile and train the model using the generator function
    train_generator = generator(X_train,y_train, batch_size=BATCH_SIZE)
    validation_generator = generator(X_test,y_test, batch_size=BATCH_SIZE)


    # Build Convolutional Neural Network in Keras
    model = Sequential()
    
    #preprocessing using Lambda layer - normalize and mean
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(32, 32,3)))
    
    
    #first Conv layer with relu and maxpool
    model.add(Convolution2D(4, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    
    #second Conv layer with relu and maxpool
    model.add(Convolution2D(8, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    
    #Third Conv layer with relu
    model.add(Convolution2D(12, 3, 3))
    model.add(Activation('relu'))


    
    #first fully connected layer follow up by dropout layer as FC layers tend to overfit
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    #second fully connected layer 
    model.add(Dense(84))
    #third fully connected layer 
    model.add(Dense(10))
    #last layer to output
    model.add(Dense(1))    
   
    model.compile(loss='mse', optimizer='adam',metrics=['mse', 'accuracy'])
    model.fit_generator(train_generator, steps_per_epoch=len(X_train)/BATCH_SIZE, validation_data=validation_generator,validation_steps=len(X_test)/BATCH_SIZE, nb_epoch=20)   
 
    X_images= extract_images(X_train)
   
    metrics = model.evaluate(X_images, y_train)
    print('')
    print(np.ravel(model.predict(X_images)))
    print(model.metrics_names)
    print(metrics)
    #for i in range(len(model.metrics_names)):
    #    print(str(model.metrics_names[i]) + ": " + str(metrics[i]))

    if(debug_save_model==1):
        model.save(model_filename)
else:
    print("==== loading NN model...")
    from keras.models import load_model
    model = load_model(model_filename)
    print("==== load done")

    print("==== Testing NN model...")

    X_images= extract_images(X_test)
  
    metrics = model.evaluate(X_images, y_test)

    print(model.metrics_names)
    print(metrics)
    
    print("one prediction:",model.predict(X_images[1:2]),"---label:",y_test[0])
    
    print("==== Test done")
    

