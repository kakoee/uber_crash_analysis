# developed by Mohammad Reza kakoee


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# SOME DEBUG VARIABLES for test images
test_undistort = 0
test_binary_threshold = 0
test_bird_eye_view=0
test_bird_eye_binary_threshold=0
test_main_lane_mark = 0

test_image_name='straight_lines1'

read_video=1
# end debug 

# first step is camera calibration 

# prepare object points

nx = 9#the number of inside corners in x
ny = 6#the number of inside corners in y

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('./camera_cal/*.jpg')
gray=[]

for fname in images:
    img = cv2.imread(fname)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)


# performs the camera calibration
# computing camera calibration matrix and distortion coefficients based on objpoints and imgpoints and shape
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# testing distortion correction in one image
if(test_undistort==1):
	img = cv2.imread('./camera_cal/calibration1.jpg')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	undist = cv2.undistort(img, mtx, dist, None, mtx)
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	f.tight_layout()
	ax1.imshow(img)
	ax1.set_title('Original Image', fontsize=50)
	ax2.imshow(undist)
	ax2.set_title('Undistorted Image', fontsize=50)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	f.savefig('./output_images/calibration1_pre_post.png')
## end of testing  distortion on one image


## defining pipeline to create a threshold binary image - first define all threshold functions

# 1st sobel threshol
def abs_sobel_threshold(myimg, orient='x', thresh=(0,255)):
    img = np.copy(myimg)

    # 1) Convert to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)

    # use l_channel for sobel threshold
    l_channel = hls[:,:,1]
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if(orient=='x'):
        sobel = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is >= thresh_min and <= thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

# 2nd magnitude threshold
def mag_threshold(myimg, sobel_kernel=3, mag_thresh=(30, 100)):

    img = np.copy(myimg)

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)

    # use l_channel for sobel threshold
    l_channel = hls[:,:,1]
    # 2) Take the derivative in x and y
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    abs_sobelxy = np.sqrt(np.square(sobelx)+np.square(sobely))
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    # 5) Create a mask of 1's 
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


#3rd dir threshold
def dir_threshold(myimg, sobel_kernel=15, thresh=(0.7, 1.3)):

    img = np.copy(myimg)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)

    # use l_channel for sobel threshold
    l_channel = hls[:,:,1]
    # 2) Take the derivative in x and y
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    abs_sobelx = np.sqrt(np.square(sobelx))
    abs_sobely=  np.sqrt(np.square(sobely))
    arctan2= np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(arctan2)
    binary_output[(arctan2 > thresh[0]) & (arctan2 < thresh[1])] = 1
    # 6) Return this mask as binary_output image
    return binary_output

#4th threshold on HLS color space
def hls_s_select(myimg, thresh=(100, 255)):

    img = np.copy(myimg)

    # 1) Convert to HLS color space
    hls_img= cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel

    # use s_channel for color threshold
    S=hls_img[:,:,2]
    binary_output=np.zeros_like(S)
    binary_output[(S>=thresh[0]) & (S<=thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output


#5th threshold on LUV color space
def luv_l_select(myimg, thresh=(180, 255)):
    img = np.copy(myimg)

    # 1) Convert to LUV color space
    luv_img= cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    # 2) Apply a threshold to the L channel

    # use l_channel for color threshold
    L=luv_img[:,:,0]
    binary_output=np.zeros_like(L)
    binary_output[(L>=thresh[0]) & (L<=thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output  

#6th threshold on Lab color space
def lab_b_select(myimg, thresh=(150, 200)):
    img = np.copy(myimg)

    # 1) Convert to Lab color space
    lab_img= cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    # 2) Apply a threshold to the b channel

    # use b_channel for color threshold
    B=lab_img[:,:,2]
    binary_output=np.zeros_like(B)
    binary_output[(B>=thresh[0]) & (B<=thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output  

    
    
    
# finally using above function to combine them and make pipeline
def combined_bin_threshold(myimg, abs_sobel_thresh=(50,100), abs_sobel_orient='x',mag_thresh=(30, 100), dir_thresh=(0.7, 1.3),s_thresh=(180, 255)):
    
    img = np.copy(myimg)

    # first undistort image    
    img= cv2.undistort(img, mtx, dist, None, mtx)

    # Threshold x gradient (on l_channel)
    sx_binary = abs_sobel_threshold(img, orient=abs_sobel_orient, thresh=abs_sobel_thresh)
    
    # mag threshold
    magbinary = mag_threshold(img, sobel_kernel=3, mag_thresh=mag_thresh)

    # dir threshold
    dirbinary = dir_threshold(img, sobel_kernel=15, thresh=dir_thresh)

    # Threshold S color channel on HLS
    s_binary = hls_s_select(img, thresh=s_thresh)

    # Threshold L color channel on LUV
    l_binary = luv_l_select(img)

    # Threshold B color channel on LAB
    b_binary = lab_b_select(img)



    combined_binary = np.zeros_like(s_binary)
    #combined_binary[(s_binary == 1) | (magbinary == 1) | (l_binary == 1) | (b_binary == 1) ] = 1
    combined_binary[  (l_binary == 1) | (b_binary == 1)   ] = 1

    #nonzero = combined_binary.nonzero()
    #if(np.array(nonzero[0]).size<20):
    #    combined_binary[  (s_binary == 1)  ] = 1

    
    
    return combined_binary



# tesing the binary threshold
if(test_binary_threshold==1):
	image = mpimg.imread('./test_images/'+test_image_name+'.jpg')
	result = combined_bin_threshold(image)
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	f.tight_layout()
	ax1.imshow(image)
	ax1.set_title('Original image', fontsize=40)
	ax2.imshow(result,cmap='gray')
	ax2.set_title('combined binary threshold', fontsize=40)
	f.savefig('./output_images/'+test_image_name+'_binary_threshold.png')
# End tesing the binary threshold



# defining function to create wrap and bird eye view
def bird_eye_view(img, mtx, dist):

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # 1) Undistort using mtx and dist
    #gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    gray = undist
    rect_src = np.zeros((4, 2), dtype = "float32")
    #rect_src[0] = [574,446]
    #rect_src[1] = [717,446]
    #rect_src[2] = [40,720]
    #rect_src[3] = [1250,720]
    
    rect_src[0] = [257,240]
    rect_src[1] = [374,240]
    rect_src[2] = [125,313]
    rect_src[3] = [506,313]
    src = np.float32(rect_src)
    rect_dist = np.zeros((4, 2), dtype = "float32") 
    rect_dist[0] = [125,47]
    rect_dist[1] = [554,47]
    rect_dist[2] = [125,313]
    rect_dist[3] = [554,313]
    dst = np.float32(rect_dist)
    img_size= (img.shape[1],img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return (warped,undist)


# defining function to create unwrap 
def DeWarp(img, mtx, dist):

    rect_src = np.zeros((4, 2), dtype = "float32")
    #rect_src[0] = [574,446]
    #rect_src[1] = [717,446]
    
    rect_src[0] = [257,240]
    rect_src[1] = [374,240]
    rect_src[2] = [125,313]
    rect_src[3] = [506,313]
    src = np.float32(rect_src)
    rect_dist = np.zeros((4, 2), dtype = "float32") 
    rect_dist[0] = [125,47]
    rect_dist[1] = [554,47]
    rect_dist[2] = [125,313]
    rect_dist[3] = [554,313]
    
    dst = np.float32(rect_dist)
    img_size= (img.shape[1],img.shape[0])
    M = cv2.getPerspectiveTransform(dst, src)
    dewarped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return dewarped


#test bird eye view
if(test_bird_eye_view==1):
	image = mpimg.imread('./test_images/'+test_image_name+'.jpg')
	binary_warped,undist_img = bird_eye_view(image, mtx, dist)
	#result = combined_bin_threshold(binary_warped) 
	#img=result
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	f.tight_layout()
	ax1.imshow(image,cmap='gray')
	ax1.set_title('Original Image', fontsize=50)
	ax2.imshow(binary_warped,cmap='gray')
	ax2.set_title('Undistorted and Warped Image', fontsize=50)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	f.savefig('./output_images/'+test_image_name+'_bird_eye_view.png')
# end test bird eye view



# tesing the binary threshold
if(test_bird_eye_binary_threshold==1):
	result = combined_bin_threshold(binary_warped)
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	f.tight_layout()
	ax1.imshow(binary_warped)
	ax1.set_title('bird_eye_view image', fontsize=40)
	ax2.imshow(result,cmap='gray')
	ax2.set_title('bird_eye binary threshold', fontsize=40)
	f.savefig('./output_images/'+test_image_name+'_bird_eye_binary_threshold.png')
# End tesing the binary threshold




#global variables: left and right poly fit
random_search=1 # to indicate first frame
left_fit =np.array([0.,0.,0.])
right_fit =np.array([0.,0.,0.])
left_lane_inds = np.array([])
right_lane_inds = np.array([])
line_distance=0
right_curverad_prev=0
left_curverad_prev=0
leftx_prev=0
lefty_prev=0
rightx_prev=0
righty_prev=0

left_fit_mov_avg=np.array([0.,0.,0.])
right_fit_mov_avg=np.array([0.,0.,0.])
left_fit_delta=0
right_fit_delta=0
left_droped=0
right_droped=0

init=1


def lane_sanity_check(myimg,new_left_fit,new_right_fit,new_left_lane_inds,new_right_lane_inds):

	global right_fit
	global left_fit
	global left_fit_mov_avg
	global right_fit_mov_avg
	global random_search
	global left_droped
	global right_droped
	global left_fit_delta
	global right_fit_delta
	
	nonzero = myimg.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 
	
	
	# calculate delta from moving average . get sum of all dimensions as approximate. better function could be written. but, this works for this project
	left_fit_delta = np.sum(np.abs(new_left_fit - left_fit_mov_avg)).astype(int)
	right_fit_delta = np.sum(np.abs(new_right_fit - right_fit_mov_avg)).astype(int)	

	Drop_left_fit=0
	Drop_right_fit=0
	left_lane_check=1
	right_lane_check=1
	
	# if delta more than 50 and less than 15 frame droped (after 15 it tries to use fIT unless the delta is more than 300)
	if((((left_fit_delta>50) and (left_droped<15)) or ((left_fit_delta>300) and (left_droped<40)))):
		Drop_left_fit=1
		left_droped+=1

		
	if((((right_fit_delta>50) and (right_droped<15)) or ((right_fit_delta>300) and (right_droped<40)))):
		Drop_right_fit=1
		right_droped+=1

	
	if((leftx.size==0) or (lefty.size==0) or (Drop_left_fit==1)):
		left_lane_check=0
	else:
		left_droped=0
	

	if((rightx.size==0)	or (righty.size==0) or (Drop_right_fit==1)):
		right_lane_check=0
	else:
		right_droped=0
	
	
	return (left_lane_check,right_lane_check)
	




def scratch_search(myimg):

	global right_fit
	global left_fit
	global left_lane_inds
	global right_lane_inds
	
	
	new_left_fit = left_fit
	new_right_fit =right_fit
	
	img=myimg
	# now we can find lanes by getting histogram on binary warped image
	# Take a histogram of the bottom half of the image
	histogram = np.sum(img[(int)(img.shape[0]/2):,:], axis=0)
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint
	
	new_left_lane_inds = []
	new_right_lane_inds = []
	nwindows = 9
	# Set height of windows
	window_height = np.int(img.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 25
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices


	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = img.shape[0] - (window+1)*window_height
		win_y_high = img.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		new_left_lane_inds.append(good_left_inds)
		new_right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
	# Concatenate the arrays of indices
	new_left_lane_inds = np.concatenate(new_left_lane_inds)
	new_right_lane_inds = np.concatenate(new_right_lane_inds)
	# Extract left and right line pixel positions
	leftx = nonzerox[new_left_lane_inds]
	lefty = nonzeroy[new_left_lane_inds] 
	rightx = nonzerox[new_right_lane_inds]
	righty = nonzeroy[new_right_lane_inds] 
	# Fit a second order polynomial to each
	#if(leftx.size ==0 | rightx.size==0):
	#	return myimg
	if(leftx.size>0):
		new_left_fit = np.polyfit(lefty, leftx, 2)
	if(rightx.size>0):
		new_right_fit = np.polyfit(righty, rightx, 2)

	return (new_left_fit,new_right_fit, new_left_lane_inds,new_right_lane_inds)
	
def directed_search(myimg,left_fit,right_fit):

	global left_lane_inds
	global right_lane_inds
	
	img=myimg
	nonzero = img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 10
	new_left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
	left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
	left_fit[1]*nonzeroy + left_fit[2] + margin))) 
	new_right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
	right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
	right_fit[1]*nonzeroy + right_fit[2] + margin)))  
	# Again, extract left and right line pixel positions
	leftx = nonzerox[new_left_lane_inds]
	lefty = nonzeroy[new_left_lane_inds] 
	rightx = nonzerox[new_right_lane_inds]
	righty = nonzeroy[new_right_lane_inds]
	# Fit a second order polynomial to each
	if(leftx.size>0):
		new_left_fit = np.polyfit(lefty, leftx, 2)
	else:
		new_left_fit = left_fit

	if(rightx.size>0):
		new_right_fit = np.polyfit(righty, rightx, 2)
	else:
		new_right_fit = right_fit

	return (new_left_fit,new_right_fit, new_left_lane_inds,new_right_lane_inds)
	

import matplotlib.image as mpimg
cnt=0

def Adv_Lane_detect_image(myimg):
	global random_search
	global right_fit
	global left_fit
	global left_lane_inds
	global right_lane_inds
	global leftx_prev
	global lefty_prev
	global rightx_prev
	global righty_prev
	global right_curverad_prev
	global left_curverad_prev
	
	global left_fit_mov_avg
	global right_fit_mov_avg
	global left_droped
	global right_droped
	global init
	global cnt
	

	# first, get bird eye view and undistort image
	binary_warped,undist_img = bird_eye_view(myimg, mtx, dist)

	# second, get binary threshold 
	result = combined_bin_threshold(binary_warped) 
	img=result
    
	#return binary_warped

	#plt.imsave("output/out"+str(cnt)+".png", img)
	
	#cnt+=1
    

	# now we can find lanes by getting histogram on binary warped image
	# Take a histogram of the bottom half of the image
	histogram = np.sum(img[(int)(img.shape[0]/2):,:], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((img, img, img))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint
	

	nonzero = img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	
	dist_top_diff=0
	dist_bottom_diff =0

	if(random_search==1): 

		new_left_fit,new_right_fit, left_lane_inds,right_lane_inds = scratch_search(img)
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds] 

	else: # if not first frame. use existing fit
		new_left_fit,new_right_fit, left_lane_inds,right_lane_inds = directed_search(img,left_fit,right_fit)

	left_lane_check=1
	right_lane_check=1
	
	# do santity check on the new fits
	if(init==0):
		left_lane_check,right_lane_check=lane_sanity_check(img,new_left_fit,new_right_fit,left_lane_inds,right_lane_inds)

	random_search = 1
		
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 
	
	
	prev_left_fit =np.zeros_like(new_left_fit)
	prev_right_fit =np.zeros_like(new_right_fit)
	
	
	#if sanity check fails, use previous fit
	if(left_lane_check==0):
		leftx = leftx_prev
		lefty = lefty_prev
		random_search = 1
	#else update fit as well as fit moving average. give more weight to recent fits
	else:
		prev_left_fit=left_fit
		left_fit = new_left_fit
		left_droped=0
		left_fit_mov_avg = (3*left_fit+left_fit_mov_avg)/4

	if(right_lane_check==0):
		rightx=rightx_prev
		righty=righty_prev
		random_search = 1
	else:
		prev_right_fit=right_fit
		right_fit = new_right_fit
		right_droped=0
		right_fit_mov_avg = (3*right_fit+right_fit_mov_avg)/4

		
	leftx_prev = leftx
	lefty_prev = lefty
	rightx_prev = rightx
	righty_prev = righty

	
	

	# Generate x and y values for plotting
	ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Create an image to draw on and an image to show the selection window
	out_img = np.dstack((img, img, img))*255
	window_img = np.zeros_like(out_img)


	left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	right_line = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])

	#draw polygon on image
	points = np.hstack((left_line, right_line))
	cv2.fillPoly(window_img, np.int_([points]), (0,255,0))

	#Shade lane area
	DeWrap = DeWarp(window_img, mtx, dist)

	#apply lane mark on undistorted image as it is calculated based on the undistorted image
	result = cv2.addWeighted(undist_img, 1, DeWrap, 0.3, 0)



	#In order to accurately convert from pixels to meters, these conversion factors need to be set appropriately based on the lane in the warped images.
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/920 # meters per pixel in x dimension
	

	y_eval = np.max(ploty)
	
	# Fit new polynomials to x,y in world space
	right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
	left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	
	left_curverad_prev = left_curverad
	right_curverad_prev=right_curverad
	# Now our radius of curvature is in meters
	
        # finding offset of car from center of the lane. 
	left_f = np.poly1d(left_fit)
	left_f1=left_f(y_eval)
	right_f = np.poly1d(right_fit)
	right_f1=right_f(y_eval)
	offset = (left_f1+ right_f1)/2.0 - myimg.shape[1]/2.0
	offset_m = xm_per_pix*offset

	Text= "Radius of Curvature : {0:.2f}m".format((left_curverad+right_curverad)/2)
	Text2 = "Offset from Center: {0:.2f}m".format(offset_m)
	#cv2.putText(result,Text, (100,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0))
	#cv2.putText(result,Text2, (100,130), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0))
	
	# for test
	Text3 = "Sanity check(Fit): {0:.2f} {1:0.2f} ".format(left_fit_delta,right_fit_delta)
	#cv2.putText(result,Text3, (100,330), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0))

	init=0
	return result



#testing main function 
if(test_main_lane_mark==1):
	image = mpimg.imread('./test_images/'+test_image_name+'.jpg')
	result=Adv_Lane_detect_image(image)
	plt.imsave('./output_images/'+test_image_name+'_lane_marked.png',result)
# end testing main func

random_search=1
init=1

## run on video
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
video_name='uber_crash'
if(read_video==1):
	white_output = video_name+'_lane_marked.mp4'
	clip1 = VideoFileClip(video_name+".mp4")#.subclip(0,5)
	white_clip = clip1.fl_image(Adv_Lane_detect_image)
	white_clip.write_videofile(white_output, audio=False)








