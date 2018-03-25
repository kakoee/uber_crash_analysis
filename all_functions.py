import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label
import sklearn


def extract_images(image_file_list):
    images = []
    for line in image_file_list:

        # adding center image and steering meas
        img_filename=line
        image = cv2.imread(img_filename)
        #print(line[0],line[1])

        #converting image to RGB as cv2 return BGR while drive.py uses RGB    
        imgRGB64=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        imgRGB = cv2.resize(imgRGB64, (0,0), fx=0.5, fy=0.5) 
        images.append(imgRGB)
    X_train = np.array(images)
    return X_train
    


#generator function 
def generator(samplesX, samplesY, batch_size=32):
    num_samples = samplesY.shape[0]

    samples_m= np.column_stack((samplesX,samplesY))

    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples_m)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            targets = []
            for line in batch_samples:

                # adding center image and steering meas
                img_filename=line[0]
                image = cv2.imread(img_filename)
                #print(line[0],line[1])

                #converting image to RGB as cv2 return BGR while drive.py uses RGB    
                imgRGB64=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                imgRGB = cv2.resize(imgRGB64, (0,0), fx=0.5, fy=0.5) 
                images.append(imgRGB)
                target = line[1]
                targets.append(target)    
    
                # adding flipped version of center image and -1*steering meas
                image_flipped = np.fliplr(imgRGB)
                target_flipped = target
                images.append(image_flipped)
                targets.append(target_flipped)

            X_train = np.array(images)
            y_train = np.array(targets)
            yield sklearn.utils.shuffle(X_train, y_train)



 
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, ystart,xstart, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        top_left = ((int)(bbox[0][0]+xstart),(int)(bbox[0][1]+ystart))
        window_x = abs(((bbox[1][0] - bbox[0][0]) )).astype(int)
        window_y = abs(((bbox[1][1] - bbox[0][1]) )).astype(int)        
        bottom_right = ((int)(bbox[1][0]+xstart),(int)(bbox[1][1]+ystart))
        
        cv2.rectangle(img, top_left,bottom_right, (0,0,255), 6)
    # Return the image
    return img

 
    
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, color_space, ystart, ystop, xstart,scales, model, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,heat_threshold):
    
    draw_img = np.copy(img)
    #img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    if color_space != 'RGB':
            if color_space == 'HSV':
                ctrans_tosearch_orig = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':                        
                ctrans_tosearch_orig = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':                        
                ctrans_tosearch_orig = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':                        
                ctrans_tosearch_orig = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':                      
                ctrans_tosearch_orig = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: 
            ctrans_tosearch_orig = np.copy(img_tosearch)    
    
    box_list = []  
    new_heat = np.zeros_like(img_tosearch[:,:,0]).astype(np.float)
    for scale in scales:
        for i in range(0,2):# repeat on shifted image on x
            #for j in range(0,2):# repeat on shifted image on y
                ctrans_tosearch = np.copy(ctrans_tosearch_orig[i*4:,i*4:])
                if scale != 1:
                    imshape = ctrans_tosearch.shape
                    ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
                    
                ch1 = ctrans_tosearch[:,:,0]
                ch2 = ctrans_tosearch[:,:,1]
                ch3 = ctrans_tosearch[:,:,2]
            
                # Define blocks and steps as above
                nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
                nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
                nfeat_per_block = orient*cell_per_block**2
                
                # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
                window = 32
                nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
                cells_per_step = 1  # Instead of overlap, define how many cells to step
                nxsteps = (int)((nxblocks - nblocks_per_window) // cells_per_step + 1)
                nysteps = (int)((nyblocks - nblocks_per_window) // cells_per_step + 1)
                
            
        
                #heat = np.zeros_like(ctrans_tosearch[:,:,0]).astype(np.float)
            
                for xb in range(nxsteps):
                    for yb in range(nysteps):
                        ypos = (int)(yb*cells_per_step)
                        xpos = (int)(xb*cells_per_step)
            
                        xleft = xpos*pix_per_cell
                        ytop = ypos*pix_per_cell
            
                        # Extract the image patch
                        subimg = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]
                        #print(subimg.shape)
                        img_array= np.array([subimg])
                        #print(img_array.shape)
        
                        test_prediction = model.predict(img_array)
                        #print(test_prediction[0])
                        
                        if ((test_prediction[0] >= 0.75)):# and abs(decision_func)>0.4):
                            #print(abs(decision_func))
                            xbox_left = np.int(xleft*scale)+i*4
                            ytop_draw = np.int(ytop*scale)+i*4
                            win_draw = np.int(window*scale)
                            box_list.append(((xbox_left, ytop_draw),(xbox_left+win_draw, ytop_draw+win_draw)))
            
 
    new_heat = add_heat(new_heat,box_list)

    new_heat = apply_threshold(new_heat,heat_threshold)
       
    return new_heat   
  
def draw_hit_map(myimg,ystart,xstart,heat):
    heatmap = np.clip(heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(myimg, ystart,xstart, labels)
    return draw_img

    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


    
    
# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


#define a funciton to crop NxN images from a bigger image
def crop_images(imgfile,window=64):
    img = cv2.imread(imgfile)
    Num_y = (int)(img.shape[0]/window)
    Num_x = (int)(img.shape[1]/window)
    images=[]
    for j in range(0,Num_y):
        for i in range(0,Num_x):
            top_left= (j*window,i*window)
            crop_img = img[j*window:((j+1)*window), i*window:((i+1)*window)]
            images.append(crop_img)
 
    img_shift= img[2:,2:]
    Num_y = (int)(img_shift.shape[0]/window)
    Num_x = (int)(img_shift.shape[1]/window)
    for j in range(0,Num_y):
        for i in range(0,Num_x):
            top_left= (j*window,i*window)
            crop_img = img_shift[j*window:((j+1)*window), i*window:((i+1)*window)]
            images.append(crop_img)
    img_shift= img[4:,4:]
    Num_y = (int)(img_shift.shape[0]/window)
    Num_x = (int)(img_shift.shape[1]/window)
    for j in range(0,Num_y):
        for i in range(0,Num_x):
            top_left= (j*window,i*window)
            crop_img = img_shift[j*window:((j+1)*window), i*window:((i+1)*window)]
            images.append(crop_img)

    return images






