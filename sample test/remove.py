# to remove water mark using coordinates
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

image = cv2.imread("/home/ceinfo/Desktop/zebra/FCOM60808042021113325.jpg")
image[:, :, 2] = 0
canny = cv2.Canny(image, 150, 250, 3)
lines = cv2.HoughLinesP(canny, cv2.HOUGH_PROBABILISTIC, np.pi/180, 20)
for line in lines:
    for x0, y0, x1, y1 in line:
        cv2.line(image, (x0,y0), (x1,y1), (255, 255, 255), 1)

fig = plt.figure(figsize = (15, 10))
fig.add_subplot(1, 2, 1).set_title("canny")
plt.imshow(canny, cmap = "gray")
fig.add_subplot(1, 2, 2).set_title("lines")
plt.imshow(image)
plt.show()




"""------------------------------------------------"""

#masking for green color

img = cv2.imread('/home/ceinfo/Desktop/zebra/FCOM60808042021113325.jpg')
red = img[:, :, 2] # to segment out red area
green = img[:, :, 1] # to segment out green are
ret, thresh1 = cv2.threshold(red, 5, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(green, 5, 255, cv2.THRESH_BINARY)
_, cnts1, _ = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
_, cnts2, _ = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
c1 = max(cnts1, key = cv2.contourArea)
c2 = max(cnts2, key = cv2.contourArea)
rect1 = cv2.minAreaRect(c1)
rect2 = cv2.minAreaRect(c2)
box1 = cv2.boxPoints(rect1)
box2 = cv2.boxPoints(rect2)
box1 = np.int0(box1)
box2 = np.int0(box2)
cv2.drawContours(img, [box1], 0, (0, 255, 255), 2)
cv2.drawContours(img, [box2], 0, (0, 255, 255), 2)
(p1, p2, p3, p4) = box1 # Unpacking tuple
h1 = (((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5) # calculating width by calculating distance
w1 = (((p2[0]-p3[0])**2 + (p2[1]-p3[1])**2)**0.5) # calculating height by calculating distance
(p1, p2, p3, p4) = box2 # Unpacking tuple
h2 = (((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5) # calculating width by calculating distance
w2 = (((p2[0]-p3[0])**2 + (p2[1]-p3[1])**2)**0.5) # calculating height by calculating distance
rofh = h2/h1
rofw = w2/w1
print("ratio of height = ", rofh, "and ratio by width = ", rofw)
plt.imshow(img)
plt.show()
