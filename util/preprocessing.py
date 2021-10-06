import glob, os, errno
import cv2
from PIL import Image

def make_square(im, fill_color=(255, 255, 255)):
    x, y = im.size
    size = max(x, y)
    new_im = Image.new('RGB', (int(size/0.7), int(size/0.7)), fill_color)
    new_im.paste(im, (int((size/0.7 - x) / 2), int((size/0.7 - y) / 2)))
    new_im = new_im.resize((128,128))

    return new_im


for fil in glob.glob("*.jpg"):

  img = Image.open(fil)

  w, h = img.size
  x1 = 1000
  y1 = 1000
  x2 = 0
  y2 = 0

  for i in range(w):
     for j in range(h):
         if img.getpixel((i, j)) != 255:
            if i < x1:
               x1 = i
            if j < y1:
               y1 = j

  for i in range(w):
     for j in range(h):
         if img.getpixel((w-1-i, h-1-j)) != 255:
            if (w-1-i) > x2:
               x2 = w-1-i
            if (h-1-j) >y2:
               y2 = h-1-j
         
  print("x1=%d y1=%d"%(x1,y1))
  print(img.getpixel((x1, y1)))

  print("x2=%d y2=%d"%(x2,y2))
  print(img.getpixel((x2, y2)))

  crop_img = img.crop((x1,y1,x2,y2))
  aligned = make_square(crop_img) 
  aligned.save('output/%s'%(fil))

