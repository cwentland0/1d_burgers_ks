import numpy as np
import matplotlib.pyplot as plt

dataDir = './Data/trainingData/'
basisDir = './Data/PODBasis/'

calcFlag = True
plotFlag = False

mu_1_vals = 4.25 + (1.25/9.)*np.linspace(0,9,10)
mu_2_vals = 0.015 + (0.015/7)*np.linspace(0,7,8)
MU_1, MU_2 = np.meshgrid(mu_1_vals,mu_2_vals)
MU_1_vec = MU_1.flatten(order='F')
MU_2_vec = MU_2.flatten(order='F')

if calcFlag:
	for paramIter,mu_1 in enumerate(MU_1_vec):
		mu_2 = MU_2_vec[paramIter]
		outputLabel = 'u_burgers_mu1_'+str(mu_1)+'_mu2_'+str(mu_2)+'_FOM'
		dataLoad = np.load(dataDir+outputLabel+'.npy')
		if (paramIter == 0):
			dataFull = dataLoad
		else: 
			dataFull = np.append(dataFull,dataLoad,axis=1)

	# dataFull -= 1. # data centering about initial condition

	# # append zero vector to induce IC consistency
	# zeroVec = np.zeros((256,1),dtype=np.float64)
	# dataFull = np.append(dataFull,zeroVec,axis=1)

	# # scale data
	dataTrainMin = np.amin(dataFull)
	dataTrainMax = np.amax(dataFull)
	normData = np.array([dataTrainMin, dataTrainMax])
	# dataFull = (dataFull - dataTrainMin)/(dataTrainMax - dataTrainMin)

	modelLabel = input("Input model label: ")

	AAT = np.dot(dataFull,dataFull.T)

	s_vals, u_vecs = np.linalg.eig(AAT) 

	idx = s_vals.argsort()[::-1] 
	s_vals = s_vals[idx]
	u_vecs = u_vecs[:,idx] 
	s_vals = np.sqrt(s_vals)
	
	np.save(basisDir+'normData_'+modelLabel+'.npy',normData)
	np.save(basisDir+'podBasis_'+modelLabel+'.npy',u_vecs)
	np.save(basisDir+'podSVals_'+modelLabel+'.npy',s_vals)

if plotFlag:
	if 's_vals' not in locals():
		s_vals = np.load(basisDir+'podSVals.npy')

	numVals = s_vals.shape[0]
	sumSq = np.sum(np.square(s_vals))
	energy = np.zeros(numVals,dtype=np.float64)
	for i in range(numVals):
		energy[i] = 100. - 100.*np.sum(np.square(s_vals[:(i+1)]))/sumSq

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.semilogy(np.linspace(1,numVals,numVals),energy)
	ax.set_ylabel('POD Residual Energy')
	ax.set_xlabel('Mode #')
	plt.savefig('./Images/podEnergyDecay.png')

