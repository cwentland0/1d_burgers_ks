import numpy as np
from keras.layers import Input, Conv1D, Dense, Flatten, Reshape, UpSampling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
import math
import os
import h5py

# get data
dataDir = './Data/trainingData/'

valPerc = 0.1
romSize = 10

mu_1_vals = 4.25 + (1.25/9.)*np.linspace(0,9,10)
mu_2_vals = 0.015 + (0.015/7)*np.linspace(0,7,8)
MU_1, MU_2 = np.meshgrid(mu_1_vals,mu_2_vals)
MU_1_vec = MU_1.flatten(order='F')
MU_2_vec = MU_2.flatten(order='F')

for paramIter,mu_1 in enumerate(MU_1_vec):
	mu_2 = MU_2_vec[paramIter]
	outputLabel = 'u_burgers_mu1_'+str(mu_1)+'_mu2_'+str(mu_2)+'_FOM'
	dataLoad = np.load(dataDir+outputLabel+'.npy')
	if (paramIter == 0):
		dataFull = dataLoad
	else:
		dataFull = np.append(dataFull,dataLoad,axis=1)

dataFull = np.reshape(dataFull,(-1,256))

# data centering about initial condition
dataFull += -1.

dataTrain, dataVal, _, _ = train_test_split(dataFull,dataFull,test_size=valPerc,random_state=7)

# scale data
dataTrainMin = np.amin(dataTrain)
dataTrainMax = np.amax(dataTrain)
dataTrain_scaled = (dataTrain - dataTrainMin)/(dataTrainMax - dataTrainMin)
dataValMin = np.amin(dataVal)
dataValMax = np.amax(dataVal)
dataVal_scaled = (dataVal - dataValMin)/(dataValMax - dataValMin)
np.save('./Data/normData.npy',np.array([dataTrainMin,dataTrainMax]))

numConvLayers = 4
numDenseLayers = 1

learn_rate = 1.e-4
decay_rate = 0.
batchSize = 20
maxEpochs = 1000
earlyStopEpochs = 100
initializationDist = 'glorot_uniform'
loss_func = mean_squared_error

encodeFilterList = [8,16,32,64]
encodeKernelList = [25,25,25,25]
encodeStrideList = [2,4,4,4]
encodeActivationList = ['elu','elu','elu','elu']
decodeFilterList = [32,16,8,1]
decodeKernelList = [25,25,25,25]
decodeStrideList = [4,4,4,2]
decodeActivationList = ['elu','elu','elu','elu']

decodeDenseInputSize = int(256/np.prod(encodeStrideList))

denseActivation = 'elu'


def CAE():
	input1 = Input(shape=(dataTrain_scaled.shape[1],),name='inputEncode')

	x = Reshape((dataTrain_scaled.shape[1],1),name='reshapeInput')(input1)

	x = Conv1D(filters=encodeFilterList[0],kernel_size=encodeKernelList[0],strides=encodeStrideList[0],padding='same',activation=encodeActivationList[0],
					kernel_initializer=initializationDist,bias_initializer='zeros',name='conv0')(x)
	for convNum in range(1,numConvLayers):
		x = Conv1D(filters=encodeFilterList[convNum],kernel_size=encodeKernelList[convNum],strides=encodeStrideList[convNum],padding='same',activation=encodeActivationList[convNum],
					kernel_initializer=initializationDist,bias_initializer='zeros',name='conv'+str(convNum))(x)

	x = Flatten(name='flatten')(x)

	x = Dense(romSize,activation=denseActivation,kernel_initializer=initializationDist,bias_initializer='zeros',name='fcnConv')(x)

	x = Dense(decodeDenseInputSize*encodeFilterList[-1],activation=denseActivation,kernel_initializer=initializationDist,bias_initializer='zeros',name='fcnDeconv')(x)

	x = Reshape(target_shape=(decodeDenseInputSize,encodeFilterList[-1]),name='reshapeConv')(x)

	x = UpSampling1D(decodeStrideList[0],name='upsamp0')(x)
	x = Conv1D(filters=decodeFilterList[0],kernel_size=decodeKernelList[0],padding='same',activation=decodeActivationList[0],
					kernel_initializer=initializationDist,bias_initializer='zeros',name='deconv0')(x)
	for deconvNum in range(1,numConvLayers):
		x = UpSampling1D(decodeStrideList[deconvNum],name='upsamp'+str(deconvNum))(x)
		# if (deconvNum == (numConvLayers-1)):
		# 	output1 = Conv1D(filters=decodeFilterList[deconvNum],kernel_size=decodeKernelList[deconvNum],padding='same',activation=decodeActivationList[deconvNum],
		# 			kernel_initializer=initializationDist,bias_initializer='zeros',name='deconv'+str(deconvNum))(x)
		# else:
		x = Conv1D(filters=decodeFilterList[deconvNum],kernel_size=decodeKernelList[deconvNum],padding='same',activation=decodeActivationList[deconvNum],
					kernel_initializer=initializationDist,bias_initializer='zeros',name='deconv'+str(deconvNum))(x)

	output1 = Reshape(target_shape=(dataTrain_scaled.shape[1],),name='reshapeOutput')(x)

	model = Model(input1,output1)

	return model

CAEModel = CAE()
CAEModel.summary()

opt_func = Adam(lr=learn_rate,decay=decay_rate)
CAEModel.compile(optimizer=opt_func,loss=loss_func)

earlyStop = EarlyStopping(patience=earlyStopEpochs)
CAEModel.fit(x=dataTrain_scaled,y=dataTrain_scaled,batch_size=batchSize,epochs=maxEpochs,
	validation_data=(dataVal_scaled,dataVal_scaled),verbose=1,callbacks=[earlyStop])

if not os.path.exists('./Models'): os.makedirs('./Models')
modelName = './Models/model_test.h5'
CAEModel.save(modelName)
