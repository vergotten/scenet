import numpy as np
import cv2
import json
import os
from keras.utils.np_utils import to_categorical

def loadOriginalImagesIntoNumpy(path, imageFiles, batch_size):
    """
    This func loads original images into numpy array.
    Returns X_train data.
    """
    
    X = np.empty((batch_size, 256, 256, 3))

    for i, image in enumerate(imageFiles): # image - path to image
        
        img = cv2.imread(path + image, 1)
        img = np.float32(cv2.resize(img, (256, 256))) / 127.5 - 1      
        
        X[i,] = img
        
    return X

def loadLabels(path, imageFiles, batch_size):
    """
    This func loads label images (with scaling).
    Returns to_categorical y_train data.
    """
    
    y = np.empty((batch_size, 256, 256))

    for i, image in enumerate(imageFiles): # image - path to image
        
        img = cv2.imread(path + image, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (256, 256))
        
        # level classification, 1000 - 1 meter
        img[img<=1000]= 0                  
        img[(img>=1000)&(img<1250)]= 1        
        img[(img>=1250)&(img<1500)]= 2        
        img[(img>=1500)&(img<1750)]= 3       
        img[(img>=1750)&(img<2000)]= 4        
        img[(img>=2000)&(img<2250)]= 5        
        img[(img>=2250)&(img<2500)]= 6       
        img[(img>=2500)&(img<2750)]= 7      
        img[(img>=2750)&(img<3000)]= 8      
        img[(img>=3000)&(img<3250)]= 9     
        img[(img>=3250)&(img<3500)]= 10    
        img[(img>=3500)&(img<3750)]= 11     
        img[(img>=3750)&(img<4000)]= 12     
        img[(img>=4000)&(img<4250)]= 13      
        img[(img>=4250)&(img<4500)]= 14      
        img[(img>=4500)&(img<4750)]= 15      
        img[(img>=4750)&(img<5000)]= 16      
        img[img>=5000] = 17     
    
        y[i,] = img.astype(int)
        
    return to_categorical(y, 18)

def imageBatchTrainGenerator(batch_size, num_folders): # imageFiles, imageLabels - lists with full imgs paths
    """
    This func is a generator to obtain X_train and y_train data per batch_size. 
    num_folders - how much folders to train (of 18625)
    
    """
    
    with open("E:/SceneNet/train_dirs.txt", 'r') as f:
        train_dirs = json.loads(f.read())
    
    for folder in train_dirs:
        imageFiles = os.listdir(folder + "photo" + "/")
        imageLabels = os.listdir(folder + "depth" + "/")
        
        batches = len(imageFiles) // batch_size
        
        if len(imageFiles) % batch_size > 0:
            batches += 1
            
        for b in range(batches):
            start = b * batch_size
            end = (b+1) * batch_size
    
            X_train = loadOriginalImagesIntoNumpy(folder + "photo" + "/", imageFiles[start:end], batch_size) # list_IDs_temp as argument
            Y_train = loadLabels(folder + "depth" + "/", imageLabels[start:end], batch_size)

        yield X_train, Y_train
        
def imageBatchValGenerator(batch_size, num_folders): # imageFiles, imageLabels - lists with full imgs paths
    """
    This func is a generator to obtain X_train and y_train data per batch_size. 
    
    """
    
    with open("E:/SceneNet/val_dirs.txt", 'r') as f:
        val_dirs = json.loads(f.read())
    
    for folder in val_dirs:
        imageFiles = os.listdir(folder + "photo" + "/")
        imageLabels = os.listdir(folder + "depth" + "/")
        
        batches = len(imageFiles) // batch_size

        if len(imageFiles) % batch_size > 0:
            batches += 1
            
        for b in range(batches):
            start = b * batch_size
            end = (b+1) * batch_size
    
            X_test = loadOriginalImagesIntoNumpy(folder + "photo" + "/", imageFiles[start:end], batch_size) # list_IDs_temp as argument
            Y_test = loadLabels(folder + "depth" + "/", imageLabels[start:end], batch_size)

        yield X_test, Y_test