#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


# first = cv2.imread("C:\\Users\\rcc\\Downloads\\road.jpg",1)
second = cv2.imread("C:\\Users\\rcc\\Downloads\\birds.webp",1)
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(second)

sharp = cv2.Laplacian(second, cv2.CV_64F)
# cv2.imwrite('Laplacian image.jpg', sharp)
plt.subplot(1,2,2)
plt.title("Sharpening")
plt.imshow(sharp)
plt.show()

# # cv2.imshow("Ayaaz", img)
# img1 = cv2.resize(first, (700,700))
# img2 = cv2.resize(second, (700,700))
# add = cv2.add(img1, img2)
# # subt = cv2.subtract(img1, img2)


# In[85]:


image = cv2.imread("C:\\Users\\rcc\\Downloads\\birds.webp",1)
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image)
# cv2.imshow("Ayaaz", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Two methods medianblur and gaussianblur
# filtered = cv2.medianBlur(image, 25)
filtered = cv2.GaussianBlur(image, (25, 25), 0)
plt.subplot(1,2,2)
plt.title("Filtered")
plt.imshow(filtered)
plt.show()
# cv2.imshow("Filtered", filtered)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[4]:


image = cv2.imread("C:\\Users\\rcc\\Downloads\\birds.webp",1)
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image)

image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

image[:,:,0] = image[:,:,0] * 0.9 #HUE
image[:,:,1] = image[:,:,1] * 1.7 #SATURATION
image[:,:,2] = image[:,:,2] * 0.5 #VALUE

Enhanced = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
plt.subplot(1,2,2)
plt.title("Enhanced")
plt.imshow(Enhanced)
plt.show()


# In[5]:


image = cv2.imread("C:\\Users\\rcc\\Downloads\\birds.webp",1)
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image)

# image1 = cv2.resize(image, None, fx = 2, fy = 2)
image1 = cv2.resize(image, (700,1000))
plt.subplot(1,2,2)
plt.title("Resized")
plt.imshow(image1)
plt.show()


# In[6]:


image = cv2.imread("C:\\Users\\rcc\\Downloads\\birds.webp",1)
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image)

image1 = 255 - image
# cv2.imshow("Ayaaz", image1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.subplot(1,2,2)
plt.title("Inverse color")
plt.imshow(image1)
plt.show()


# In[7]:


image = cv2.imread("C:\\Users\\rcc\\Downloads\\birds.webp",1)
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image1 = cv2.equalizeHist(gray_image)
plt.subplot(1,2,2)
plt.title("Equilized")
plt.imshow(image1)
plt.show()


# In[8]:


image = cv2.imread("C:\\Users\\rcc\\Downloads\\birds.webp",1)
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image)

image1 = cv2.Canny(image, 100, 200)
plt.subplot(1,2,2)
plt.title("Edges")
plt.imshow(image1)
plt.show()


# In[9]:


image = cv2.imread("C:\\Users\\rcc\\Downloads\\birds.webp")
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image)
image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
log_value = 100 * np.log(1 + image2)
# cv2.imshow("Ayaaz", log_value)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.subplot(1,2,2)
plt.title("Original")
plt.imshow(log_value)
plt.show()


# In[10]:


# Assignment Problem:


image = cv2.imread("C:\\Users\\rcc\\Downloads\\road.jpg",1)
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image)
# plt.show()

# Gray Image
image = cv2.imread("C:\\Users\\rcc\\Downloads\\road.jpg",1)
image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
res = cv2.resize(image2, (500, 500))
plt.subplot(1,2,2)
# cv2.imshow("Gray", res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.subplot(1,2,2)
plt.title("Gray")
plt.imshow(res)
plt.show()

# x = int(input("Enter the value of X: "))
# y = int(input("Enter the value of Y: "))
x = 10
y = 20
four = [(x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)]
plt.subplot(1,2,1)
plt.title("Four Neighbours")
plt.imshow(four)
# plt.show()

eight = [(x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1),(x - 1, y - 1), (x - 1, y + 1), (x + 1, y + 1), (x + 1, y - 1) ]
plt.subplot(1,2,2)
plt.title("Eight Neighbours")
plt.imshow(eight)
plt.show()
# print("For finding Distance..")
# x1 = int(input("Enter the value of X1: "))
# y1 = int(input("Enter the value of Y1: "))
# x2 = int(input("Enter the value of X2: "))
# y2 = int(input("Enter the value of Y2: "))

# distance = np.sqrt(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
# print("\nThe distance is: ",distance)
# # plt.subplot(1,2,2)
# plt.title("Distance..")
# plt.imshow(distance)

# Negative Image



negative_image = 255 - image2
res = cv2.resize(negative_image, (500, 500))
plt.subplot(1,2,2)
plt.title("Negative Image")
plt.imshow(res)
plt.show()
# cv2.imshow("Negative Image", res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# image = cv2.imread("C:\\Users\\rcc\\Downloads\\birds.webp")
# plt.subplot(1,2,1)
# plt.title("Original")
# plt.imshow(image)
# plt.show()

# image3 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
log_value = 100 * np.log(1 + image2)
plt.subplot(1,2,1)
plt.title("Log transformation")
plt.imshow(log_value)
# plt.show()

# image = cv2.imread("C:\\Users\\rcc\\Downloads\\birds.webp")
# plt.subplot(1,2,1)
# plt.title("Original")
# plt.imshow(image)

# image4 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
power_transfer = np.power(image2, 2.8)
plt.subplot(1,2,2)
plt.title("Power Law Transformation")
plt.imshow(power_transfer)
plt.show()


# In[11]:


img = cv2.imread("C:\\Users\\rcc\\Downloads\\road.jpg")
image = cv2.resize(img, (500, 500))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

negative_image = 255 - image
# res = cv2.resize(negative_image, (500, 500))

log_value = 50 * np.log(1 + image2)

power_transfer = np.power(image2, 0.5)

cv2.imshow("Original", image)
cv2.imshow("Negative Image", negative_image.astype(np.uint8))
cv2.imshow("Power", power_transfer.astype(np.uint8))
cv2.imshow("Log", log_value.astype(np.uint8))


cv2.waitKey(0)
cv2.destroyAllWindows()



# In[ ]:





# In[ ]:





# In[12]:


# Log transformation

image = cv2.imread("C:\\Users\\rcc\\Downloads\\birds.webp")
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image)
image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
log_value = 100 * np.log(1 + image2)
# cv2.imshow("Ayaaz", log_value)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.subplot(1,2,2)
plt.title("Log transformation")
plt.imshow(log_value)
plt.show()


# In[13]:


image = cv2.imread("C:\\Users\\rcc\\Downloads\\birds.webp")
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image)
image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
power_transfer = np.power(image2, 2.8)
# cv2.imshow("Ayaaz", log_value)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.subplot(1,2,2)
plt.title("Power Law Transformation")
plt.imshow(power_transfer)
plt.show()


# In[14]:


# cv2.imshow("Ayaaz", add)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[15]:


import cv2
import numpy as np

# Load image
image = cv2.imread('C:\\Users\\rcc\\Downloads\\birds.webp', cv2.IMREAD_GRAYSCALE)

# Define functions

def four_neighbours(x, y):
    return [(x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)]

def eight_neighbours(x, y):
    return [(x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1),
            (x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1), (x + 1, y + 1)]

def distance_formula(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def image_negation(img):
    max_pixel_value = np.max(img)
    negated_img = max_pixel_value - img
    return negated_img

def log_transformation(img):
    log_transformed_img = np.log1p(img)
    return log_transformed_img

def power_law_transformation(img, gamma):
    power_transformed_img = np.power(img, gamma)
    return power_transformed_img

# Example Usage

# Choose a pixel position (x, y)
x, y = 50, 50

# Neighbours
four_neighbors = four_neighbours(x, y)
eight_neighbors = eight_neighbours(x, y)

# Distance between two points
x1, y1 = 10, 10
x2, y2 = 20, 20
distance = distance_formula(x1, y1, x2, y2)

# Image operations
negated_image = image_negation(image)
log_transformed_image = log_transformation(image)
gamma_value = 0.5  # You can adjust this value
power_law_transformed_image = power_law_transformation(image, gamma_value)

# Display images (for example)
cv2.imshow('Original Image', image)
cv2.imshow('Negated Image', negated_image.astype(np.uint8))
cv2.imshow('Log Transformed Image', log_transformed_image.astype(np.uint8))
cv2.imshow('Power Law Transformed Image', power_law_transformed_image.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[66]:


img = cv2.imread("C:\\Users\\rcc\\Downloads\\road.jpg",0)

image = cv2.resize(img, (500, 500))
cv2.imshow("ayaaz", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[102]:


img = cv2.imread("C:\\Users\\rcc\\Downloads\\birds.webp",0)

h, w = img.shape
print(h, w)
a = int(input("Enter the value of a: "))
b = int(input("Enter the value of b: "))

for i in range(h):
    for j in range(w):
        if (img[i][j] < a or img[i][j] > b):
            img[i][j] = 10
            
        else:
            img[i][j] = 100

cv2.imshow("ayaaz", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




