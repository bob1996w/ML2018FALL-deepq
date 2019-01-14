# when running:
# PYTHONHASHSEED=0 python3 train.py
# PYTHONHASHSEED=0 python3 train.py <ModelName without .h5> <Training DataPath> <Image DataPath> <Model Structure> <epoch> <seed>
import os
import sys

pathName = sys.argv[1]
trainRawPath = sys.argv[2]
imageRootPath = sys.argv[3]
model_structure = sys.argv[4]
epoch_to_train = int(sys.argv[5])
seed_number = sys.argv[6]

import random
random.seed(int(seed_number))

import numpy as np
np.random.seed(int(seed_number))

import tensorflow as tf
tf.set_random_seed(int(seed_number))

import keras
from keras import backend as K
sess = tf.Session(graph=tf.get_default_graph())
K.set_session(sess)


import pandas as pd
from keras.utils import to_categorical, np_utils
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, CSVLogger
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.densenet import preprocess_input
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, LeakyReLU, Activation, GlobalMaxPooling2D, Input
from keras.utils.generic_utils import CustomObjectScope
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.datasets import make_classification
from keras.optimizers import Adam



def SplitDataToTrainAndValidation(trainRawPath, trainPath, validationPath, seed, validationNum):  #切training set 和 validation set
	trainRawData = pd.read_csv(trainRawPath, encoding='big5', dtype='O')
	trainRawLebel = trainRawData['Labels'].str.split(' ')

	labelIndex = []
	for dataIndex in range(len(trainRawLebel)):
		if type(trainRawLebel[dataIndex]) == list:
			labelIndex.append(dataIndex)
	random.Random(seed).shuffle(labelIndex)
	#print(labelIndex)

	trainData = []
	validationData = []
	for i in range(len(labelIndex)):
		if i < validationNum:
			validationData.append(list(trainRawData.loc[labelIndex[i]]))
		else:
			trainData.append(list(trainRawData.loc[labelIndex[i]]))

	trainData = pd.DataFrame(trainData, columns=list(trainRawData))
	validationData = pd.DataFrame(validationData, columns=list(trainRawData))
	#print(trainData)
	
	trainData.to_csv(trainPath, index=False)
	validationData.to_csv(validationPath, index=False)

def ReadImage(imagePath, imageDim=224):
	image = load_img(imagePath)
	image = image.resize(size=(imageDim, imageDim))
	return img_to_array(image)

def PreprocessTrainData(X, Y, datagen, imgDatagen, imageDim=224):
	X = np.array(X)
	Y = np.array(Y)
	datagen.fit(X)

	X_gen = []
	Y_gen = []
	## use "datagen" correctly?  num = (batch_size * imgDatagen)
	batches = 0
	for x_batch, y_batch in datagen.flow(X, Y):
		batches = batches + 1
		for x in x_batch:
			X_gen.append(x)
		for y in y_batch:
			Y_gen.append(y)
		if batches == imgDatagen:
			break
	# set weight (neccessary?)
	W = []
	for y in Y_gen:
		w = 1
		#for label in y:
		#	if yy[index][label_num] != 0:
		#		w = max(w,label_weight[label_num])
		W.append(w)
	return preprocess_input(np.array(X_gen).reshape(-1, imageDim, imageDim, 3)), np.array(Y_gen), np.array(W)

def GetTrainData(path, imageRootPath, batch_size, imgDatagen=4):
	trainFile = open(path,'r')
	trainRawData = trainFile.readlines()
	trainFile.close()
	
	trainDataSize = len(trainRawData)-1

	imageName = []
	imageLabel = []

	for rowIndex in range(1, trainDataSize+1):
		trainCol = trainRawData[rowIndex].strip().split(',')
		imageName.append(trainCol[0])
		imageLabel.append(np.array(trainCol[1].split(' ')).astype(float))

	datagen = ImageDataGenerator(rotation_range=5.0)

	while True:
		cnt = 0
		X = []
		Y = []
		for index in range(trainDataSize):
			X.append(ReadImage(os.path.join(imageRootPath, imageName[index])))
			Y.append(imageLabel[index])
			cnt = cnt + 1
			if cnt == batch_size:
				yield PreprocessTrainData(X, Y, datagen, imgDatagen)
				cnt = 0
				X = []
				Y = []
		if cnt != 0:
			yield PreprocessTrainData(X, Y, datagen, imgDatagen)

def PreprocessValidationData(X, Y, imageDim=224):
	return preprocess_input(np.array(X).reshape(-1, imageDim, imageDim, 3)), np.array(Y)

def GetValidationData(path, imageRootPath, batch_size):
	validationFile = open(path,'r')
	validationRawData = validationFile.readlines()
	validationFile.close()
	
	validationDataSize = len(validationRawData)-1

	imageName = []
	imageLabel = []

	for rowIndex in range(1, validationDataSize+1):
		validateCol = validationRawData[rowIndex].strip().split(',')
		imageName.append(validateCol[0])
		imageLabel.append(np.array(validateCol[1].split(' ')).astype(float))

	while True:
		cnt = 0
		X = []
		Y = []
		for index in range(validationDataSize):
			X.append(ReadImage(os.path.join(imageRootPath, imageName[index])))
			Y.append(imageLabel[index])
			cnt = cnt + 1
			if cnt == batch_size:
				yield PreprocessValidationData(X, Y)
				cnt = 0
				X = []
				Y = []
		if cnt != 0:
			yield PreprocessValidationData(X, Y)

def my_auc_roc(model, validationPath):
	validationFile = open(validationPath,'r')
	validationRawData = validationFile.readlines()
	validationFile.close()
	
	validationDataSize = len(validationRawData)-1

	imageName = []
	imageLabel = []

	for rowIndex in range(1, validationDataSize+1):
		validateCol = validationRawData[rowIndex].strip().split(',')
		imageName.append(validateCol[0])
		imageLabel.append(np.array(validateCol[1].split(' ')).astype(float))
	
	pro= model.predict_generator(generator=GetValidationData(validationPath, imageRootPath, batch_size["validation"]),
								steps=(validationNum // batch_size["validation"]))
	imageLabel = np.array(imageLabel)
	pro = np.array(pro)
	roc_auc_record = []
	for i in range(len(pro[0])):
		roc_auc_record.append(roc_auc_score(imageLabel[:,i],pro[:,i]))
	
	avg = 0
	for record in roc_auc_record:
		avg += record
	avg /= len(roc_auc_record)
	roc_auc_record.append(avg)
		
	return roc_auc_record


#####################
### Create folder ###
#####################

if not os.path.exists('dir_'+pathName):
	os.makedirs('dir_'+pathName)
else:
	print("The folder named \"" +"dir_"+ pathName + "\" has exist!")
	exit()

#################
### Parameter ###
#################
seed = seed_number   # 6, 2
validationNum = 1500
imageDim = 224
epoch_count = 0
max_auc_roc = 0.0
min_loss = 999
if model_structure == '121':
	batch_size = {"train": 8, "validation": 50}
else:
	batch_size = {"train": 6, "validation": 50}
#label_weight = [9,35,9,5,20,15,85,25,20,40,45,60,40,200]
label_weight = [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
model_count = 1

##################
### Split data ###
##################

trainPath = os.path.join("dir_"+pathName, 'train_file.csv')
validationPath = os.path.join("dir_"+pathName, 'validation_file.csv')
SplitDataToTrainAndValidation(trainRawPath, trainPath, validationPath, seed, validationNum)

###################
### Build Model ###
###################
model = None

if model_structure == '121':
	densenet_121_base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(imageDim, imageDim, 3))
	# add a global spatial average pooling layer
	x = densenet_121_base_model.output
	x = GlobalAveragePooling2D()(x)
	# x = MaxPooling2D(pool_size=(2, 2))(x)
	# x = Conv2D(filters=1024, kernel_size=3, padding='same', activation='relu')(x)
	# x = MaxPooling2D(pool_size=(2, 2))(x)
	# x = Dropout(0.6)(x)
	# x = Flatten()(x)
	# x = Dropout(0.6)(x)
	# and a logistic layer
	predictions = Dense(14, activation="sigmoid")(x)
	# this is the model we will train
	model = Model(inputs=densenet_121_base_model.input, outputs=predictions)
elif model_structure == '169':
	densenet_169_base_model = DenseNet169(include_top=False, weights='imagenet', input_shape=(imageDim, imageDim, 3))
	x = densenet_169_base_model.output
	x = GlobalAveragePooling2D()(x)
	predictions = Dense(14, activation="sigmoid")(x)
	# this is the model we will train
	model = Model(inputs=densenet_169_base_model.input, outputs=predictions)
elif model_structure == '201':
	densenet_201_base_model = DenseNet201(include_top=False, weights='imagenet', input_shape=(imageDim, imageDim, 3))
	x = densenet_201_base_model.output
	x = GlobalAveragePooling2D()(x)
	predictions = Dense(14, activation="sigmoid")(x)
	# this is the model we will train
	model = Model(inputs=densenet_201_base_model.input, outputs=predictions)
else:
	print('error model structure : '+model_structure)
	exit()



#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[])

#mp = sys.argv[4]
#model = keras.models.load_model(mp)





##########################
### Callback Functions ###
##########################

logPath = os.path.join("dir_"+pathName, 'training.log')

callback = []

csv_logger = CSVLogger(logPath)
checkpoint = ModelCheckpoint(filepath=pathName+'.h5', verbose=1, save_best_only=True, monitor='val_acc', mode='max')

#saveBestModel = ModelCheckpoint(model_name + '.h5', monitor="val_loss", save_best_only=True, save_weights_only=False)
# callback for evaluating every K epochs
class evaluateOnKEpochs(keras.callbacks.Callback):
	def __init__(self, K=5, **kwargs):
		self.K = K
		self.evaluate_args = kwargs
		#print(self.evaluate_args)
	def on_epoch_end(self, epoch, logs={}):
		global min_loss
		global max_auc_roc
		global seed_number
		global epoch_count
		if epoch % self.K == 0:
			print('=====loss=====')
			evaluate_data = self.model.evaluate_generator(**self.evaluate_args)
			print(evaluate_data)
		if epoch % self.K == 0:
			print('=====auc_roc=====')
			auc_roc_record = my_auc_roc(self.model, validationPath)
			print(auc_roc_record)

my_callbacks = [csv_logger,
				evaluateOnKEpochs(K=1, generator=GetValidationData(validationPath, imageRootPath, batch_size["validation"]),
								steps=(validationNum // batch_size["validation"]))]



#########################
###  Start  Training  ###
#########################
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-5, amsgrad=False)
'''

for layer in densenet_121_base_model.layers:
	layer.trainable = False
'''
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[])

print(model.summary())

trainHistory = model.fit_generator(
	GetTrainData(trainPath, imageRootPath, batch_size["train"]),
	steps_per_epoch=(10001-validationNum)//batch_size["train"],
	epochs=epoch_to_train,
	verbose=1, 
	callbacks=my_callbacks)



##########################
###    Save   Model    ###
##########################
print('Save Model')
model.save(pathName+'.h5')
