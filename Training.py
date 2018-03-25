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
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
import pickle
#from sklearn.cross_validation import train_test_split

## debug
debug_train=1
debug_save_model =1
debug_custom_train_extract=1
##



#1st Training

images_notped = glob.glob('./Train_dataset/nocar_bike_ped/*.png')
images_pedcar = glob.glob('./Train_dataset/ped_bike/*.png')



cars = []
notcars = []
for image in images_notped:
    notcars.append(image)
    
for image in images_pedcar:
    cars.append(image)

np.random.shuffle(cars) 
np.random.shuffle(notcars)

notcars=notcars[0:(int)(len(cars)/2)]

 
#custom_train_car
images_cars_custom = glob.glob('./Train_dataset/custom_ped/*.png')
for image in images_cars_custom:
    cars.append(image)

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
    notcars.append(image)

 
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (8, 8) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [350, 700] # Min and max in y to search in slide_window()
xstart=0

#model_filename = 'finalized_model_GridSearch_YCrCb.sav'
model_filename = 'finalized_model_rbf_YCrCb.sav'
scaler_filename = 'finalized_scaler.std'



if(debug_train==1):
    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)
        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)    
    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use SVM with gridsearch 
#    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svr = svm.SVC(kernel='rbf',C=1)
    
#   svc = GridSearchCV(svr, parameters)
    svc=svr
    #svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    print("training: fit started")
    svc.fit(X_train, y_train)
    print(svc)    
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    if(debug_save_model==1):
        pickle.dump(svc, open(model_filename, 'wb'))  
        pickle.dump(X_scaler, open(scaler_filename, 'wb'))         
else:
    print("loading SVM model and standardScaler...")
    svc = pickle.load(open(model_filename, 'rb'))
    X_scaler = pickle.load(open(scaler_filename, 'rb'))
    print("load done")
