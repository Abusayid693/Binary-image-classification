from PIL import Image
from scipy import ndimage
import numpy as np 
import matplotlib.pyplot as plt
import copy

num_positive=103
num_negative=76
num_px=150;
"""
Initial preparation of datasets

image: Holds all image datasets in the format [num_px,num_px,channel]
    
train_set_arr : Holds all image datasets in vectorised formats
    
train_set : numpy array to hold training set data in the format [features,number_of_training_examples]
  
"""

image_positive=[]
train_set_arr_positive=[]
image_negative=[]
train_set_arr_negative=[]

image_positive,train_set_arr_positive=loadDatasets("positive",num_positive)
image_negative,train_set_arr_negative=loadDatasets("negative",num_negative)

""" Displaying images in matplotlib """ 

plt.imshow(image_negative[1])
plt.show()    
 
""" Preparing positive datasets """ 

train_set_positive = np.asarray(train_set_arr_positive)
train_set_positive=train_set_positive.reshape(train_set_positive.shape[0],-1).T

""" Preparing negative datasets """ 
train_set_negative = np.asarray(train_set_arr_negative)
train_set_negative=train_set_negative.reshape(train_set_negative.shape[0],-1).T

X=np.zeros((67500,num_positive+num_negative))
Y=np.zeros((num_positive+num_negative,1))

X_,Y=prepareDataSets(train_set_positive,train_set_negative)



"""
   Compute the length of vectors 
"""

print(train_set_positive.shape)
print(train_set_negative.shape)
print(X.shape) 
print(Y.shape) 


"""
Preprocessing :
    To represent color images, the red, green and blue channels (RGB) must be specified
    for each pixel, and so the pixel value is actually a vector of three numbers ranging
    from 0 to 255, to normalize we divide by 255(max-value)
"""
X=X_/255

""" Displaying images in matplotlib """ 
plt.imshow(X_[:,12].reshape((num_px, num_px, 3)))
plt.show() 

print(Y[0,12])

