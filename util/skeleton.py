import cv2
import glob, os, errno
from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
import numpy as np

mydir = r'output'

element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))

for fil in glob.glob("*.jpg"):
    image = cv2.imread(fil,0) 
    image = invert(image)
    ret,image = cv2.threshold(image, 127, 255, 0)

    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)

    while True:
        #Step 2: Open the image
        open = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv2.subtract(image, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(image, element)
        skel = cv2.bitwise_or(skel,temp)
        image = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(image)==0:
            break

    skel = invert(skel)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to greyscale    

    cv2.imwrite(os.path.join(mydir,fil),skel) # write to location with same name
