import numpy as np
import sys
import os
import pandas as pd
import keras
import random
from keras.utils import to_categorical, np_utils
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, CSVLogger
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.densenet import preprocess_input
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,GlobalAveragePooling2D, BatchNormalization,LeakyReLU,Activation, GlobalMaxPooling2D,Input
#import cv2
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.datasets import make_classification

from keras.utils.generic_utils import CustomObjectScope

image_size = 224
model_path = sys.argv[1]
output_path = sys.argv[2]
test_path = sys.argv[3]
img_folder_path = sys.argv[4] # end with '/'

def ReadImage(imagePath, imageDim=224):
    image = load_img(imagePath)
    image = image.resize(size=(imageDim, imageDim))
    return img_to_array(image)

columns = ['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia',
 'Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia']

print('')
print('model_path:', model_path)
print('output_path:', output_path)
print('Predicting...')
#with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
#    model = keras.models.load_model(model_name+'.h5')
model = keras.models.load_model(model_path)

# test_path = './test.csv'
test_data = pd.read_csv(test_path)
predict = []
batch_size = 1000
for i in range(0, len(test_data), batch_size):
    batch_data = []
    real_len = len(test_data[i:i+batch_size])
    for j in range(i, i + real_len):
        img = ReadImage(img_folder_path+test_data['Image Index'][j])
        batch_data.append(img)
    batch_data = np.array(batch_data)
    r1_img = batch_data.reshape(-1,image_size,image_size,3)
    r2_img = preprocess_input(r1_img)
    batch_predict = model.predict(r2_img)
    for j in range(real_len):
        predict.append(batch_predict[j])
    if i % 100 == 0:
        print(i, '/', len(test_data))
predict = np.array(predict)
result = pd.DataFrame({'Id': test_data['Image Index']})
#print(result)
for d in range(len(columns)):
    print(columns[d], predict[:,d])
    result[columns[d]] = predict[:, d]
result.to_csv(output_path, index=False)
