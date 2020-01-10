import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from annFuncs import extractEncoderDecoder, extractJacobian, arrayScaling

podBasisLoc = './Data/PODBasis/podBasis.npy'
fomDataLoc = './Data/u_burgers_mu1_4.3_mu2_0.021_FOM.npy'
modelLoc = './Models/model_test_save.h5'
normLoc = './Data/normData.npy'
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

encoder, decoder = extractEncoderDecoder(modelLoc,N,numConvLayers,romSize,
                                	encodeKernelList,encodeFilterList,encodeStrideList,encodeActivationList,
                                	decodeKernelList,decodeFilterList,decodeStrideList,decodeActivationList,
                                	decodeDenseInputSize,denseActivation,initDist)

# code = np.random.random(romSize)
# jacob = extractJacobian(decoder,code)

fomSol = np.load(fomDataLoc)
N, numSamps = fomSol.shape

normData = np.load(normLoc)

uProj_save = np.zeros((N,numSamps),dtype=np.float32)
for i in range(numSamps):
	u = fomSol[:,i]
	u = arrayScaling(u,normData,1)
	# u = np.reshape(u,(1,-1))
	code = np.squeeze(encoder.predict(np.array([u,])))
	uProj = np.squeeze(decoder.predict(np.array([code,])))
	uProj = arrayScaling(uProj,normData,-1)
	uProj_save[:,i] = uProj

t = np.linspace(0.,35.,numSamps)
x = np.linspace(0.,10.,N)
X, T = np.meshgrid(x,t)
figContourPlot = plt.figure()
axContourPlot = figContourPlot.add_subplot(111)
levels = np.linspace(0.6,6.0,10)
ctr = axContourPlot.contourf(X,T,uProj_save.T,cmap=cm.viridis,levels=levels)
axContourPlot.set_xlabel('x')
axContourPlot.set_ylabel('t')
plt.colorbar(ctr)
plt.savefig('./Images/contour_projCAE.png')


VMat_load = np.load(podBasisLoc)
VMat = VMat_load[:,:romSize]
uProj_pod = np.dot(VMat,np.dot(VMat.T,fomSol))
figContourPlot_pod = plt.figure()
axContourPlot_pod = figContourPlot_pod.add_subplot(111)
ctr_pod = axContourPlot_pod.contourf(X,T,uProj_pod.T,cmap=cm.viridis,levels=levels)
axContourPlot.set_xlabel('x')
axContourPlot.set_ylabel('t')
plt.colorbar(ctr_pod)
plt.savefig('./Images/contour_projPOD.png')