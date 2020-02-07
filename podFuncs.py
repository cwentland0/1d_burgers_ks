import numpy as np 
import os 
import matplotlib.pyplot as plt
from annFuncs import scaleOp, invScaleOp


def evalPODProj(uSol,u0,podBasis,normData):
	uEval = scaleOp(uSol - u0, normData)
	code = np.dot(podBasis.T,uEval.T).T
	return code 

def evalPODLift(code,u0,podBasis,normData): 
	uEval = np.dot(podBasis,code.T).T
	uSol = invScaleOp(uEval, normData) + u0
	return uSol 

def evalPODFullProj(uSol,u0,podBasis,normData): 
	uEval = scaleOp(uSol - u0, normData)
	code = np.dot(podBasis.T,uEval.T).T
	uEval = np.dot(podBasis,code.T).T
	uSol = invScaleOp(uEval, normData) + u0
	return uSol

def computeProjField_pod(podBasis,fomSolLoc,u0,normDataLoc,plotFlag,saveCode):
	fomSol = np.load(fomSolLoc) 
	normData = np.load(normDataLoc)

	numPoints, numSamps = fomSol.shape

	codeSol = np.zeros((podBasis.shape[1],numSamps),dtype=np.float64)	
	fomSol_proj = np.zeros(fomSol.shape,dtype=np.float64) 

	u0_rep = np.tile(np.reshape(u0,(numPoints,1)),(1,numSamps))
	codeSol = np.dot(podBasis.T,scaleOp(fomSol-u0_rep,normData))
	fomSol_proj = invScaleOp(np.dot(podBasis,codeSol),normData) + u0_rep

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
		np.save('./Data/'+codeSaveName+'.npy',codeSol)

# def computeTCNTrainingData_POD(podBasisLoc,fomSolLoc,)