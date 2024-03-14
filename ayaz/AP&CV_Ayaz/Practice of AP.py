#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


arr = np.arange(1,10)
arr


# In[3]:


a1 = np.array([[1,2,3],[4,5,6]])
a1


# In[4]:


a1 = np.array([2,3,4])
a2 = np.array([3,4,5])
l = a1 + a2


# In[5]:


l


# In[6]:


slicing = l[0:3]
slicing


# In[7]:


addition = np.add([2,3,4],[3,4,5])
addition


# In[8]:


substraction = np.subtract([3,4,5],[2,3,4])
substraction


# In[9]:


a = np.array([[[1,2,3],[4,5,6],[7,8,9]]])
b = np.array([[[9,8,7],[6,5,4],[3,2,1]]])
addition = np.add(a,b)
addition


# In[10]:


slicing = a[0:2][1:4]
slicing


# In[11]:


addition[1:1]


# In[12]:


a = np.array([[1,2,3],[4,5,6]])
a[0:2]


# In[13]:


a = np.array([[1,2,3],[4,5,6]])
a[0:2,1:3]


# In[14]:


a = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[2,2,3],[5,5,6],[8,8,9]]])
a[0,:,1]


# In[15]:


a[1,0,1:]


# In[16]:


a = np.arange(0,30)
a[0::5]


# In[17]:


np.linspace(0,3,10)


# In[18]:


np.linspace(0,1)


# In[19]:


a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
a.shape


# In[20]:


a.reshape(1,12)


# In[21]:


np.eye(3)


# In[22]:


np.identity(3)


# In[23]:


np.zeros(3)


# In[24]:


zeros = np.array([0,1,2,3,4,5,6,7,8,9])
zeros


# In[25]:


vowels = np.array(['a','e','i','o','u'])
vowels


# In[26]:


ones = np.array([[1,1,1,1,1],[1,1,1,1,1]])
ones.shape


# In[27]:


myarray1 = np.array([[2.7,-2,-19],[0,3.4,99.9],[10.6,0,13]])
myarray1.shape


# In[28]:


myarray1


# In[29]:


myarray2 = np.arange(4.,64,4)
myarray2


# In[30]:


myarray2.shape


# In[31]:


myarray2.reshape(3,5)


# In[ ]:





# In[ ]:




