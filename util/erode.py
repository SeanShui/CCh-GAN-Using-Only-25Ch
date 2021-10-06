import cv2
import glob, os, errno
from skimage.morphology import skeletonize
from skimage import io,data
import matplotlib.pyplot as plt
from skimage.util import invert
import numpy as np

mydir = r'output'

kernel = np.ones((2, 2), np.uint8)

for fil in glob.glob("*.png"):
    image = io.imread(fil)
  
    for i, v1 in enumerate(image):
        for j, v2 in enumerate(v1):
            if (j<40) and (i<40):
                image[i, j] = 255

    erosion = cv2.erode(image, kernel, iterations=1)
    #dilate = cv2.dilate(image, kernel, iterations=1)

    io.imsave(os.path.join(mydir,fil),erosion)


