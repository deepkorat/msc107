#!/usr/bin/env python
# coding: utf-8

# In[95]:


import cv2
import matplotlib.pyplot as plt 
from PIL import Image as im
import numpy as np


# In[197]:


img = cv2.imread("C:\\Users\\rcc\\Downloads\\blur image.jpg",0)
plt.hist(img.ravel(),250,[10,250],color = 'C1')
plt.show()


# In[200]:


img1 = cv2.imread("C:\\Users\\rcc\\Downloads\\blur1.jpg",0)
plt.hist(img1.ravel(),250,[25,250],color = 'C3')
plt.show()


# In[106]:


img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img1)
plt.show()


# In[ ]:





# In[42]:


# Equilize Histogram

img2 = cv2.equalizeHist(img)
plt.hist(img2.ravel(),250,[0,250])
# plt.imshow(img1)
# plt.subplot(1,2,2)
plt.show()


# In[43]:


img2 = cv2.equalizeHist(img1)
plt.hist(img2.ravel(),250,[0,250])
# plt.imshow(img1)
# plt.subplot(1,2,2)
plt.show()


# In[122]:


# Dark blur image

img = cv2.imread("C:\\Users\\rcc\\Downloads\\b and w.webp",0)
plt.subplot(1,2,1)
plt.imshow(img)


img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.subplot(1,2,2)
plt.imshow(img1)


plt.show()
# img1 = cv2.resize(img,(500,500))
# cv2.imshow("Image",img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[175]:


img = cv2.imread("C:\\Users\\rcc\\Downloads\\b and w.webp",0)

plt.subplot(1,2,1)
plt.title("Histogram")
plt.hist(img.ravel(),255,[0,255], color = 'C3')

img1 = cv2.equalizeHist(img)

plt.subplot(1,2,2)
plt.title("Equilize histo")
plt.hist(img1.ravel(),255,[0,255],color = 'C5')
# plt.imshow(img)
plt.show()


# In[179]:


img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.subplot(1,2,1)
plt.hist(img1.ravel(),255,[0,255],color = 'C4')
plt.title("Histo")

img2 = cv2.equalizeHist(img)

plt.subplot(1,2,2)
plt.title("Equilize histo")
plt.hist(img2.ravel(),255,[0,255],color = 'C1')

plt.show()


# In[201]:


img = cv2.imread("C:\\Users\\rcc\\Downloads\\dark blur image.webp",0)
plt.hist(img.ravel(),255,[10,255],color = 'C2')
# plt.imshow(img)
plt.show()


# In[182]:


img2 = cv2.equalizeHist(img)
plt.hist(img2.ravel(),250,[0,250], color = 'C3')
plt.show()


# In[61]:


img = cv2.imread("C:\\Users\\rcc\\Downloads\\white blur image.webp",1)
plt.imshow(img)
plt.show()


# In[67]:


# White blur image

img = cv2.imread("C:\\Users\\rcc\\Downloads\\white blur image.webp",0)
plt.hist(img.ravel(),255,[0,255])
# plt.imshow(img)
plt.show()


# In[70]:


img2 = cv2.equalizeHist(img)
plt.hist(img2.ravel(),250,[0,250])
plt.show()


# In[107]:


img = cv2.imread("C:\\Users\\rcc\\Downloads\\blur high contrast image.jfif",1)
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img1)
plt.show()


# In[190]:


img = cv2.imread("C:\\Users\\rcc\\Downloads\\blur high contrast image.jfif",0)
plt.hist(img.ravel(),255,[10,255],color = 'C3')
# plt.imshow(img)
plt.show()


# In[194]:


img2 = cv2.equalizeHist(img)
plt.hist(img2.ravel(),250,[0,250],color = 'C4')
plt.show()


# In[92]:


img = cv2.imread("C:\\Users\\rcc\\Downloads\\blur high contrast image.jfif",1)
img1 = cv2.resize(img,(500,500))
cv2.imshow("Image",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[87]:


img = cv2.imread("C:\\Users\\rcc\\Downloads\\blur high contrast image.jfif",0)
img1 = cv2.resize(img,(500,500))
cv2.imshow("Image",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[93]:


img = cv2.imread("C:\\Users\\rcc\\Downloads\\blur high contrast image.jfif",0)
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img1)
plt.show()


# In[212]:


img = cv2.imread("C:\\Users\\rcc\\Downloads\\blur high contrast image.jfif",0)
image = cv2.resize(img,(200,200))
img1 = cv2.imread("C:\\Users\\rcc\\Downloads\\b and w.webp",0)
image1 = cv2.resize(img1,(200,200))
img2 = cv2.imread("C:\\Users\\rcc\\Downloads\\white blur image.webp",0)
image2 = cv2.resize(img2,(200,200))


# In[213]:


res = np.hstack((image,image1,image2))
cv2.imshow("res",res)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[38]:


cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Image",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




