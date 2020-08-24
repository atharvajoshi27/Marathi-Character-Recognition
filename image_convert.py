import cv2
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

model1 = load_model('devanagari_model.h5')
img = cv2.imread('test3.jpg', 1)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#plt.imshow(img)
#plt.show()
(height, width) = img_hsv.shape
flag = 0
for i in range(0, height):
    for j in range(0, width):
        if(img_hsv[i][j] == 0):
            y1 = i
            flag = 1
            break
    if(flag):
        break
flag = 0
for i in range(0, width):
    for j in range(0, height):
        if(img_hsv[j][i] == 0):
            x1 = i
            flag = 1
            break
    if(flag):
        break
flag = 0
for i in range(height-1, 0, -1):
    for j in range(width-1, 0, -1):
        if(img_hsv[i][j] == 0):
            y2 = i
            flag = 1
            break
    if(flag):
        break        
flag = 0
for i in range(width-1, 0, -1):
    for j in range(height-1, 0, -1):
        if(img_hsv[j][i] == 0):
            x2 = i
            flag = 1
            break
    if(flag):
        break
#print(y1, y2)
#print(x1, x2)
y1 = y1-int(height/16)
y2 = y2+int(height/16)
x1 = x1-int(width/16)
x2 = x2+int(width/16)
img_roi = img_hsv[y1:y2, x1:x2]
plt.imshow(img_roi)
img_pass = cv2.resize(img_roi, (32, 32))
plt.imshow(img_pass)
for i in range(0, 32):
    for j in range(0, 32):
        img_pass[i][j] = 255 - img_pass[i][j]
img_pass = np.array(img_pass, dtype=np.float32)
img_pass = np.reshape(img_pass, (-1, 32, 32, 1))
y_prob = model1.predict_classes(img_pass)
print("Predicted Class:")
print(y_prob)