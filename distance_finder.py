import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# defining function to create wrap and bird eye view
def bird_eye_view(img):

    undist = img
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

    
    rect_src[0] = [272,248]
    rect_src[1] = [355,248]
    rect_src[2] = [198,290]
    rect_src[3] = [414,290]
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


file='output/uber_crashTrim1_Moment6_w.png'
imageBGR = cv2.imread(file)

image = cv2.cvtColor(imageBGR,cv2.COLOR_BGR2RGB) 

binary_warped,undist_img = bird_eye_view(image)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(binary_warped,cmap='gray')
ax2.set_title('Birdeye view Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
f.savefig('./output/moment6_bird_eye_view.png')


