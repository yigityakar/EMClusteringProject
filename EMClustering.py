#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
from scipy.stats import multivariate_normal


# In[2]:


data_set = np.genfromtxt("hw07_data_set.csv",delimiter= ",")
initial= np.genfromtxt("hw07_initial_centroids.csv",delimiter=",")


# In[3]:


hik= np.zeros((len(data_set),len(initial)))
K=len(initial)
N=len(data_set)
#print(initial)


# In[4]:


for i in range(0,len(data_set)):

    hik[i][np.argmin(np.stack((np.linalg.norm(data_set[i]-initial[c]) for c in range(K))))]=1
    #print(np.stack((np.linalg.norm(data_set[i]-initial[c]) for c in range(0,5))))
 


# In[5]:


priors=np.stack(np.sum(hik[:,i]/len(hik)) for i in range(K))


# In[6]:


arr1=[]
arr2=[]
arr3=[]
arr4=[]
arr5=[]
arr=[arr1,arr2,arr3,arr4,arr5]

for i in range(0,N):
    arr[np.argmax(hik[i])].append(data_set[i])
    


# In[7]:


for i in range(K):
    arr[i]=np.array(arr[i])


# In[8]:


cov_list=[0,0,0,0,0]
for i in range(K):
    cov_list[i]=np.cov(np.transpose(arr[i]))
#print(cov_list)
h=[]
temp_mean3=[]


# In[9]:


for index in range(100):
    
    if index==1:
        h=hik
        h=np.array(h)
    else:
        for i in range (K):


            a= multivariate_normal.pdf(data_set,mean=initial[i],cov= cov_list[i] ) * priors[i]/ (np.sum(multivariate_normal.pdf(data_set,mean=initial[c],cov= cov_list[c]) * priors[c] for c in range(5) ))
            temp_mean3.append(a)
                          
    
        h=np.transpose(np.vstack(temp_mean3))
        temp_mean3=[]
        temp_mean=[]



    
    for i in range(K):
        for n in range(N):
            temp_mean.append(h[n][i]*data_set[n])
        
        temp_mean2=np.array(temp_mean)
        temp_mean2=np.sum(temp_mean2,axis=0)/np.sum(np.sum(h[:,i]))
    
        initial[i]=temp_mean2
        temp_mean=[]

    temp_cov=[]
    for i in range(K):
        for n in range(N):
            temp_cov.append(h[n][i] *np.matmul((data_set[n]-initial[i])[:,None], [data_set[n]-initial[i]]))
        temp_cov2=np.array(temp_cov) 
        temp_cov2=np.sum(temp_cov2,axis=0)/np.sum(h[:,i])
        cov_list[i]=temp_cov2
        temp_cov=[]
    


    priors=np.stack(np.sum(h[:,i]/len(h)) for i in range(K))
    


# In[10]:


print(initial)


# In[11]:


h1=[]
h2=[]
h3=[]
h4=[]
h5=[]

hf=[h1,h2,h3,h4,h5]


for i in range(N):
    hf[np.argmax(h[i])].append(data_set[i])
for i in range(K):
    hf[i]=np.array(hf[i])


# In[12]:


plt.figure(figsize=(10,10))
plt.scatter(hf[0][:,0], hf[0][:,1],color="blue")
plt.scatter(hf[1][:,0], hf[1][:,1],color="green")
plt.scatter(hf[2][:,0], hf[2][:,1],color="red")
plt.scatter(hf[3][:,0], hf[3][:,1],color="orange")
plt.scatter(hf[4][:,0], hf[4][:,1],color="purple")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




