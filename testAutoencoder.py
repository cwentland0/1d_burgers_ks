import numpy as np
from annFuncs import extractEncoderDecoder, extractJacobian

modelLoc = './Models/model_test_save.h5'
N = 256
numConvLayers = 4
romSize = 10

encodeFilterList = [8,16,32,64]
encodeKernelList = [25,25,25,25]
encodeStrideList = [2,4,4,4]
encodeActivationList = ['elu','elu','elu','elu']
decodeFilterList = [32,16,8,1]
decodeKernelList = [25,25,25,25]
decodeStrideList = [4,4,4,2]
decodeActivationList = ['elu','elu','elu','elu']

initDist = 'glorot_uniform'
denseActivation = 'elu'
decodeDenseInputSize = int(N/np.prod(encodeStrideList))

encoder, decoder, decoder_grad = extractEncoderDecoder(modelLoc,N,numConvLayers,romSize,
                                	encodeKernelList,encodeFilterList,encodeStrideList,encodeActivationList,
                                	decodeKernelList,decodeFilterList,decodeStrideList,decodeActivationList,
                                	decodeDenseInputSize,denseActivation,initDist)
