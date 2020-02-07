import numpy as np
from tensorflow.keras.layers import Input, Add, Conv1D, Dense, Flatten, Reshape, UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import tensorflow.keras.backend as K
from tcn import TCN
from sklearn.model_selection import train_test_split
import math
import os
import h5py

######## LOAD DATA ########
dataDir = './Data/trainingData/'

valPerc = 0.1 	# percentage set aside for validation
romSize = 10 	# code dimension

mu_1_vals = 4.25 + (1.25/9.)*np.linspace(0,9,10)
mu_2_vals = 0.015 + (0.015/7)*np.linspace(0,7,8)
MU_1, MU_2 = np.meshgrid(mu_1_vals,mu_2_vals)
MU_1_vec = MU_1.flatten(order='F')
MU_2_vec = MU_2.flatten(order='F')

for paramIter,mu_1 in enumerate(MU_1_vec):
	mu_2 = MU_2_vec[paramIter]
	outputLabel_full = 'u_burgers_mu1_'+str(mu_1)+'_mu2_'+str(mu_2)+'_FOM'
    outputLabel_code = outputLabel_full+'_code_'+num2str(romSize)
	dataLoad_full = np.load(dataDir+outputLabel_full+'.npy')
    dataLoad_code = np.load(dataDir+outputLabel_code+'.npy')
	if (paramIter == 0):
		dataFull = dataLoad_full
        dataCode = dataLoad_code
    else:
		dataFull = np.append(dataFull,dataLoad_full,axis=1)
        dataCode = np.append(dataCode,dataLoad_code,axis=1)

dataFull = np.reshape(dataFull,(-1,256))
dataCode = np.reshape(dataCode,(-1,romSize))

######## DATA STANDARDIZATION #######
dataFull -= 1. # data centering about initial condition

# append zero vector to induce IC consistency
zeroVec = np.zeros((1,256),dtype=np.float64)
dataFull = np.append(dataFull,zeroVec,axis=0)

# shuffle data to avoid biasing training to end of training
dataTrainCode, dataValCode, dataTrainFull, dataValFull = train_test_split(dataCode,dataFull,test_size=valPerc,random_state=24)

# scale data
dataTrainMin = np.amin(dataTrainFull)
dataTrainMax = np.amax(dataTrainFull)
dataTrainFull_scaled = (dataTrainFull - dataTrainMin)/(dataTrainMax - dataTrainMin)
dataValMin = np.amin(dataValFull)
dataValMax = np.amax(dataValFull)
dataValFull_scaled = (dataValFull - dataValMin)/(dataValMax - dataValMin)
# np.save('./Data/normData.npy',np.array([dataTrainMin,dataTrainMax]))


####### NETWORK PARAM SETTINGS ########
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
decodeActivationList = ['elu','elu','elu','linear']

decodeDenseInputSize = int(256/np.prod(encodeStrideList))

denseActivation = 'elu'

####### NETWORK DEFINITION #######
def l2Loss(yTrue,yPred):
    return K.sum(K.square(yTrue - yPred))

def CAE_TCN():
	input1 = Input(shape=(dataTrain_scaled.shape[1],),name='inputDecode')

	reshapedInput = Reshape((dataTrain_scaled.shape[1],1),name='reshapeInput')(input1)


	
    x = Add()([x, reshapedInput])

	x = Dense(decodeDenseInputSize*encodeFilterList[-1],activation=denseActivation,kernel_initializer=initializationDist,bias_initializer='zeros',name='fcnDeconv')(x)

	x = Reshape(target_shape=(decodeDenseInputSize,encodeFilterList[-1]),name='reshapeConv')(x)

	x = UpSampling1D(decodeStrideList[0],name='upsamp0')(x)
	x = Conv1D(filters=decodeFilterList[0],kernel_size=decodeKernelList[0],padding='same',activation=decodeActivationList[0],
					kernel_initializer=initializationDist,bias_initializer='zeros',name='deconv0')(x)
	for deconvNum in range(1,numConvLayers):
		x = UpSampling1D(decodeStrideList[deconvNum],name='upsamp'+str(deconvNum))(x)
		x = Conv1D(filters=decodeFilterList[deconvNum],kernel_size=decodeKernelList[deconvNum],padding='same',activation=decodeActivationList[deconvNum],
					kernel_initializer=initializationDist,bias_initializer='zeros',name='deconv'+str(deconvNum))(x)

	output1 = Reshape(target_shape=(dataTrain_scaled.shape[1],),name='reshapeOutput')(x)

	model = Model(input1,output1)

	return model

####### BUILD MODEL AND TRAIN #######
CAEModel = CAE_TCN()
CAEModel.summary()

opt_func = Adam(lr=learn_rate,decay=decay_rate)
CAEModel.compile(optimizer=opt_func,loss=l2Loss)

earlyStop = EarlyStopping(patience=earlyStopEpochs)
CAEModel.fit(x=dataTrain_scaled,y=dataTrain_scaled,batch_size=batchSize,epochs=maxEpochs,
	validation_data=(dataVal_scaled,dataVal_scaled),verbose=1,callbacks=[earlyStop])


####### SAVE MODEL #######
if not os.path.exists('./Models'): os.makedirs('./Models')
modelName = './Models/model_test_new.h5'
CAEModel.save(modelName)