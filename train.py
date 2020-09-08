from enet import *
from data_generator import *
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
import pandas as pd 
import os 
import cv2
import numpy as np
from keras import optimizers
from keras.utils import multi_gpu_model

n_classes = 18
model = ENET(img_height=256, img_width=256, nclasses=n_classes)

#model.summary()

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)

print("python {}".format(sys.version))
print("keras version {}".format(keras.__version__)); del keras
print("tensorflow version {}".format(tf.__version__))

warnings.filterwarnings("ignore")

#os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
model = multi_gpu_model(model, gpus=2) #in this case the number of GPus is 2 !!!!!!!!!!!!!!!!!!
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)

train_batch_size = 30
val_batch_size = 30
train_num_folders = 10
val_num_folders = 1

train_steps_per_epoch = train_num_folders*300 / train_batch_size
val_steps_per_epoch = val_num_folders*300 / val_batch_size

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse','accuracy'])

print("Training on amount of images: ", train_num_folders*300)
print("Validating on amount of images: ", val_num_folders*300)

hist = model.fit_generator(
    generator=imageBatchTrainGenerator(train_batch_size, train_num_folders), steps_per_epoch=train_steps_per_epoch, epochs=10,
    validation_data=imageBatchValGenerator(val_batch_size, val_num_folders), validation_steps=val_steps_per_epoch, verbose = 1)

model.save('E:/NN\'s/weights, data on so so/scenet322_enet.h5')

with open('file_hist_enet.json', 'w') as f:
    json.dump(hist.history, f)

for key in ['loss', 'val_loss']:
    plt.plot(hist.history[key],label=key)
plt.legend()
plt.show()

for key in ['acc', 'val_acc']:
    plt.plot(hist.history[key],label=key)
plt.legend()
plt.show()

for key in ['mean_squared_error']:
    plt.plot(hist.history[key],label=key)
plt.legend()
plt.show()