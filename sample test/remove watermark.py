import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

dir = os.getcwd()
path = "/home/ceinfo/Desktop/zebra/FCOM60808042021113325.jpg"

newPath = "sample test/removed.jpg"
img=cv2.imread(path,1)
plt.imshow(img)
plt.show()

hight,width,depth=img.shape[0:3]
print(hight)
print(width)

#Intercept
# cropped = img[int(hight*0.8):hight, int(width*0.7):width]  #The clipping coordinates are [Y0: Y1, x0: X1]

cropped = img[int(hight*0.8):hight, int(width*0.7):width]  #The clipping coordinates are [Y0: Y1, x0: X1]
plt.imshow(cropped)
plt.show()

'''576*896'''
cv2.imwrite(newPath, cropped)
imgSY = cv2.imread(newPath,1)

#Image binarization processing, changing colors other than [200200] - [250250] to 0
thresh = cv2.inRange(imgSY,np.array([200,200,200]),np.array([250,250,250]))

#Create structural elements of shapes and dimensions
kernel = np.ones((3,3),np.uint8)

#Expand the area to be repaired
hi_mask = cv2.dilate(thresh,kernel,iterations=10)
specular = cv2.inpaint(imgSY,hi_mask,5,flags=cv2.INPAINT_TELEA)
cv2.imwrite(newPath, specular)

#Overlay picture
imgSY = Image.open(newPath)
img = Image.open(path)
img.paste(imgSY, (int(width*0.7),int(hight*0.8),width,hight))
img.save(newPath)