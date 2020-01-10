import numpy as np
from keras.layers import Input, Conv1D, Dense, Flatten, Reshape, UpSampling1D
from keras.models import Model
from keras import backend as K
import tensorflow as tf

def extractJacobian(gradients,decoder,code):

	codeTensor = np.reshape(code,(1,-1))
	sess = tf.InteractiveSession()
	sess.run(tf.initialize_all_variables())
	jacob = sess.run(gradients,feed_dict={model.input:codeTensor})
	ses.close()

	return jacob	


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

	decoderJacobian = K.gradients(decoder_model.output,decoder_model.input)

	import pdb; pdb.set_trace()

	return encoder_model, decoder_model, decoderJacobian
