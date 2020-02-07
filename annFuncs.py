import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Reshape, UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import h5py
import time

def scaleOp(uSol,normData): 
	uSol = (uSol - normData[0])/(normData[1] - normData[0])
	return uSol 

def invScaleOp(uSol,normData):
	uSol = uSol*(normData[1] - normData[0]) + normData[0] 
	return uSol

def evalEncoder(uSol,u0,encoder,normData):
	uEval = scaleOp(uSol - u0, normData)
	code = np.squeeze(encoder.predict(np.array([uEval,])))
	return code 

def evalDecoder(code,u0,decoder,normData): 
	uEval = np.squeeze(decoder.predict(np.array([code,])))
	uSol = invScaleOp(uEval, normData) + u0
	return uSol

def evalCAE(uSol,u0,encoder,decoder,normData): 
	uEval = (uSol - u0 - normData[0])/(normData[1] - normData[0])
	code = np.squeeze(encoder.predict(np.array([uEval,])))
	uEval = np.squeeze(decoder.predict(np.array([code,])))
	uSol = uEval*(normData[1] - normData[0]) + u0 + normData[0]
	return uSol

def extractJacobian(decoder,code): 

	with tf.GradientTape() as g:
		inputs = tf.Variable(np.reshape(code,(1,-1)),dtype=tf.float32)
		outputs = decoder(inputs)

	jacob = np.squeeze(g.jacobian(outputs,inputs).numpy()) 

	return jacob	

def extractNumJacob(decoder,code,stepSize):
	uSol = np.squeeze(decoder.predict(np.array([code,]))) 
	numJacob = np.zeros((uSol.shape[0],code.shape[0]),dtype=np.float64)
	for elem in range(0,code.shape[0]):
		tempCode = code.copy()
		tempCode[elem] = tempCode[elem] + stepSize 
		output = np.squeeze(decoder.predict(np.array([tempCode,])))
		numJacob[:,elem] = (output - uSol).T/stepSize

	return numJacob

def computeProjField_ann(encoder,decoder,fomSolLoc,u0,normDataLoc,plotFlag,saveCode): 
	fomSol = np.load(fomSolLoc) 
	normData = np.load(normDataLoc)

	numPoints, numSamps = fomSol.shape

	codeSol = np.zeros((encoder.output.shape[1],numSamps),dtype=np.float64)	
	fomSol_proj = np.zeros(fomSol.shape,dtype=np.float64) 

	for t in range(0,numSamps):
		codeSol[:,t] = evalEncoder(fomSol[:,t],u0,encoder,normData)
		fomSol_proj[:,t] = evalDecoder(codeSol[:,t],u0,encoder,decoder,normData)


	if plotFlag:
		t = np.linspace(0,1,numSamps)  
		x = np.linspace(0,1,numPoints) 
		X,T = np.meshgrid(x,t) 

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.contourf(X,T,fomSol_proj.T) 
		plt.savefig('./Images/projField.png')

	if saveCode:
		codeSaveName = input("Input name to save latent space code history file: ")
		np.save('./Data/'+codeSaveName+'.npy')


def extractEncoderDecoder(modelLoc,N,numConvLayers,romSize,
				encodeKernelList,encodeFilterList,encodeStrideList,encodeActivationList,
				decodeKernelList,decodeFilterList,decodeStrideList,decodeActivationList,
				decodeDenseInputSize,denseActivation,initDist):

	def encoder():
		input1 = Input(shape=(N,),name='inputEncode')
		x = Reshape((N,1),name='reshapeInput')(input1)
		x = Conv1D(filters=encodeFilterList[0],kernel_size=encodeKernelList[0],strides=encodeStrideList[0],padding='same',activation=encodeActivationList[0],
                                        kernel_initializer=initDist,bias_initializer='zeros',name='conv0')(x)
        	
		for convNum in range(1,numConvLayers):
			x = Conv1D(filters=encodeFilterList[convNum],kernel_size=encodeKernelList[convNum],strides=encodeStrideList[convNum],padding='same',activation=encodeActivationList[convNum],
                                        kernel_initializer=initDist,bias_initializer='zeros',name='conv'+str(convNum))(x)

		x = Flatten(name='flatten')(x)
		output1 = Dense(romSize,activation=denseActivation,kernel_initializer=initDist,bias_initializer='zeros',name='fcnConv')(x)
		model = Model(input1,output1)

		return model



	def decoder():

		input1 = Input(shape=(romSize,),name='inputDecode')
		x = Dense(decodeDenseInputSize*encodeFilterList[-1],activation=denseActivation,kernel_initializer=initDist,bias_initializer='zeros',name='fcnDeconv')(input1)
		x = Reshape(target_shape=(decodeDenseInputSize,encodeFilterList[-1]),name='reshapeConv')(x)

		x = UpSampling1D(decodeStrideList[0],name='upsamp0')(x)
		x = Conv1D(filters=decodeFilterList[0],kernel_size=decodeKernelList[0],padding='same',activation=decodeActivationList[0],
                                        kernel_initializer=initDist,bias_initializer='zeros',name='deconv0')(x)
        
		for deconvNum in range(1,numConvLayers):
			x = UpSampling1D(decodeStrideList[deconvNum],name='upsamp'+str(deconvNum))(x)
			x = Conv1D(filters=decodeFilterList[deconvNum],kernel_size=decodeKernelList[deconvNum],padding='same',activation=decodeActivationList[deconvNum],
                                        kernel_initializer=initDist,bias_initializer='zeros',name='deconv'+str(deconvNum))(x)

		output1 = Reshape(target_shape=(N,),name='reshapeOutput')(x)
		model = Model(input1,output1)

		return model

	encoder_model = encoder()
	encoder_model.load_weights(modelLoc,by_name=True)

	decoder_model = decoder()
	decoder_model.load_weights(modelLoc,by_name=True)

	return encoder_model, decoder_model


def breakCAE(CAELoc,saveLoc,modelLabel):

	N = 256
	romSize = 10

	numConvLayers = 4
	encodeFilterList = [8,16,32,64]
	encodeKernelList = [25,25,25,25]
	encodeStrideList = [2,4,4,4]
	encodeActivationList = ['elu','elu','elu','elu']
	decodeFilterList = [32,16,8,1]
	decodeKernelList = [25,25,25,25]
	decodeStrideList = [4,4,4,2]
	decodeActivationList = ['elu','elu','elu','linear']
	initDist = 'glorot_uniform'
	denseActivation = 'elu'
	decodeDenseInputSize = int(N/np.prod(encodeStrideList))

	encoder, decoder = extractEncoderDecoder(CAELoc,N,numConvLayers,romSize,
				encodeKernelList,encodeFilterList,encodeStrideList,encodeActivationList,
				decodeKernelList,decodeFilterList,decodeStrideList,decodeActivationList,
				decodeDenseInputSize,denseActivation,initDist) 

	encoder.save(os.path.join(saveLoc,'encoder_'+modelLabel+'.h5'))
	decoder.save(os.path.join(saveLoc,'decoder_'+modelLabel+'.h5'))