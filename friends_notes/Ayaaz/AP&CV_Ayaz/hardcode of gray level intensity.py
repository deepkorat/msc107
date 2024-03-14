#!/usr/bin/env python
# coding: utf-8

# In[33]:


import cv2
import matplotlib.pyplot as plt
import numpy as np


# In[43]:


# image = cv2.imread("C:\\Users\\rcc\\Downloads\\birds.webp")
# image = cv2.imread("C:\\Users\\rcc\\Downloads\\nature.jfif")
image = cv2.imread("C:\\Users\\rcc\\Downloads\\road.jpg",0)
# image = cv2.imread("C:\\Users\\rcc\\Downloads\\mask.jpg")
image = cv2.resize(image, (500, 600))
image


# In[35]:


r1 = image.shape[0]
c1 = image.shape[1]
r1, c1


# In[27]:


#Intensity Slicing
for i in range(r):
    for j in range(c):
        if image[i][j] > 100:
            image[i][j] = 150
        elif image[i][j] < 10:
            image[i][j] = 150
        else:
            image[i][j]
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[13]:


image = cv2.imread("C:\\Users\\rcc\\Downloads\\road.jpg")
image = cv2.resize(image, (500, 600))
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
lower_thres = 100
upper_thres = 200
mask = np.where((gray_image >= lower_thres) & (gray_image <= upper_thres), 255, 0).astype(np.uint8)
sliced_image = cv2.bitwise_and(image, image, mask = mask)
cv2.imshow("Image", image)
cv2.imshow("Sliced Image", sliced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[50]:


# for i in range(r):
#     for j in range(c):
#         if image[i][j] >= 100:
#             image[i][j] = 150
#         elif image[i][j] <= 200:
#             image[i][j] = 0
#         else:
#             image[i][j]
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[1]:


# plt.figure(figsize = (15,15))
# plt.boxplot(image)
# plt.show()


# In[57]:


#filter2D
k = np.array([[1/9,1/9,1/9],
              [1/9,1/9,1/9],
              [1/9,1/9,1/9]])

r2, c2 = k.shape[:2]   
r2, c2


# In[58]:


filtered = np.zeros_like(image)
for i in range(r1 - r2 + 1):
    for j in range(c1 - c2 + 1):
        interest = image[i:i + r2, j:j + c2]
#         print(interest)
#         break
#         image_ = (k * interest).sum()
        image_ = np.multiply(interest, k).sum()
        filtered[i][j] = image_
        
filtered = np.clip(filtered, 0, 255)
cv2.imshow("Image", image)
cv2.imshow("filtered Image", filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[59]:


filtered = np.zeros_like(image)
filtered


# In[66]:


import numpy as np
import cv2
# Read the image in greyscale
img = cv2.imread('C:\\Users\\rcc\\Downloads\\road.jpg',0)

#Iterate over each pixel and change pixel value to binary using np.binary_repr() and store it in a list.
lst = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
         lst.append(np.binary_repr(img[i][j] ,width=8)) # width = no. of bits

# We have a list of strings where each string represents binary pixel value. To extract bit planes we need to iterate over the strings and store the characters corresponding to bit planes into lists.
# Multiply with 2^(n-1) and reshape to reconstruct the bit image.
eight_bit_img = (np.array([int(i[0]) for i in lst],dtype = np.uint8) * 128).reshape(img.shape[0],img.shape[1])
seven_bit_img = (np.array([int(i[1]) for i in lst],dtype = np.uint8) * 64).reshape(img.shape[0],img.shape[1])
six_bit_img = (np.array([int(i[2]) for i in lst],dtype = np.uint8) * 32).reshape(img.shape[0],img.shape[1])
five_bit_img = (np.array([int(i[3]) for i in lst],dtype = np.uint8) * 16).reshape(img.shape[0],img.shape[1])
four_bit_img = (np.array([int(i[4]) for i in lst],dtype = np.uint8) * 8).reshape(img.shape[0],img.shape[1])
three_bit_img = (np.array([int(i[5]) for i in lst],dtype = np.uint8) * 4).reshape(img.shape[0],img.shape[1])
two_bit_img = (np.array([int(i[6]) for i in lst],dtype = np.uint8) * 2).reshape(img.shape[0],img.shape[1])
one_bit_img = (np.array([int(i[7]) for i in lst],dtype = np.uint8) * 1).reshape(img.shape[0],img.shape[1])

#Concatenate these images for ease of display using cv2.hconcat()
finalr = cv2.hconcat([eight_bit_img,seven_bit_img,six_bit_img,five_bit_img])
finalv =cv2.hconcat([four_bit_img,three_bit_img,two_bit_img,one_bit_img])

# Vertically concatenate
final = cv2.vconcat([finalr,finalv])

# Display the images
cv2.imshow('a',final)
cv2.waitKey(0) 


# In[67]:


eight_bit_img


# In[ ]:





# In[ ]:




