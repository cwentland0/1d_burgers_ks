"""
Driver for computing the finite difference approximate solution to the Burgers' or Kuramoto-Sivashinsky equation
	Burgers': u_t = - u*u_x + nu*u_xx 
	Kuramoto-Sivashinsky: u_t = - u*u_x - u_xx - nu*u_xxxx
Author: Christopher Wentland, University of Michigan
Date: August 2019
"""

import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import numpy as np
import spaceSchemes
from timeSchemes import storeTimeDiffParams
from onlineFuncs import computeSol, computeProjSol
import tensorflow as tf
import matplotlib.pyplot as plt
from annFuncs import extractEncoderDecoder 
from tensorflow.keras.models import load_model

def main():

	if not os.path.exists('./Data'): os.makedirs('./Data')
	if not os.path.exists('./Images'): os.makedirs('./Images')
	if not os.path.exists('./RestartFiles'): os.makedirs('./RestartFiles')

	simType = 'FOM'  # 'FOM', 'PODG', 'PODG-MZ', 'PODG-TCN', 'GMan', 'GMan-TCN'
	paramBurgers = False
	calcProj = False

	# governing equation selection
	problem = 'ks'		# 'burgers': Burgers' equation, 'ks': Kuramoto-Sivashinsky equation
	
	# output parameters 
	outputDir = ''; outputLoc = os.path.join('./Data',outputDir) # where data files are written to
	if not os.path.exists(outputLoc): os.makedirs(outputLoc)

	saveSol = False
	saveRHS = False
	saveCode = False
	plotSol = True 		# plot time-history contours (can take a bit for fine grids/time sampling, long time evolution)
	plotSnaps = True  		# plot real-time line plots
	calcErr = False 
	compareType = 'u'     # either 'u' or 'RHS' for now


	# temporal and spatial derivative approximates
	# note: linear second-order and fourth-order derivatives are always computed by central schemes
	timeDiffScheme = 'BDF'				# 'BDF': backwards differentiation (implicit), 'RK': Runge-Kutta (explicit) 
	timeOrdAcc = 2						# 1-4 (first-order to fourth-order accuracy)
	nonlinDiffScheme = 'central'		# 'upwind': upwind scheme, 'central': central scheme
	nonlinOrdAcc = 2					# 1,2,4 (first-order to third-order accuracy)
										# note: upwind scheme only accepts 1; central scheme only accepts 2,4
	linOrdAcc = 2 						# 2, 4 (second-order or fourth-order accuracy)

	# parameterized Burgers'
	param_mult = 0.02 
	mu_1 = 4.3
	mu_2 = 0.021

	# boundary conditions
	bound_cond = 'periodic'
	bc_vals = [mu_1,1]

	spaceDiffParams = {'nonlinDiffScheme':nonlinDiffScheme,'nonlinOrdAcc':nonlinOrdAcc,'linOrdAcc':linOrdAcc,'bound_cond':bound_cond,'bc_vals':bc_vals}
	timeDiffParams = storeTimeDiffParams(timeDiffScheme,timeOrdAcc)

	# spatial domain settings
	N = 512							# number of spatial points for finite-difference discretization
	xi = 0.0						# x-coordinate of 'left' boundary
	xf = 32*math.pi 				# x-coordinate of 'right' (periodic) boundary
	dx = (xf-xi)/float(N-1)			# uniform distance between spatial points
	x = np.linspace(xi,xf,N) 	# vector of spatial points

	# set initial conditions
	ICType = 'mixedSinCos' 									# 'turbulent': mock-turbulent/Burgulence 
															# 'sin': sine wave
															# 'mixedSinCos': product of cosine and offset sine 
	unif_val = 1.
	randSeed = 7											# fixed seed for RNG in 'turbulence' IC
	numWaves = 12											# number of waves in 'turbulence' IC
	angFreq = 1./16. 										# angular frequency for 'sin' or mixedSinCos' ICs 
	u0 = spaceSchemes.setICs(ICType,x,unif_val,randSeed,numWaves,angFreq)	# initial condition field

	# temporal integration parameters
	restartFlag = False 					# restart from restart file
	restartFile = 'restart_ks_8000' 			# name of restart npy file, sans file extension
										# note: I just manually create restart files from solution save data
										# 	Just append the iteration number and time to the beginning of the snapshot array and save
	tEnd = 150 						# end time of simulation
	dt = 0.01 							# time step
	Nt = int(round(tEnd/dt)) 			# total number of time steps from t = 0 to tEnd
	sampRate = 10 						# time step interval for saving solution to disk

	# compute discrete linear operator matrix
	viscosity = 1.																# dissipation coefficient
	linOp, bc_vec = spaceSchemes.precompLinOp(problem,spaceDiffParams,N,dx,viscosity) 		# compute linear dissipation operator
	source_term = 0.0*x
	# source_term = param_mult*np.exp(mu_2*x)

	# param study values
	mu_1_vals = 4.25 + (1.25/9.)*np.linspace(0,9,10)
	mu_2_vals = 0.015 + (0.015/7)*np.linspace(0,7,8)
	MU_1, MU_2 = np.meshgrid(mu_1_vals,mu_2_vals)
	MU_1_vec = MU_1.flatten(order='F')
	MU_2_vec = MU_2.flatten(order='F')

	# ROM parameters 
	romSize = 10

	# ROM basis/decoders load
	VMat = []
	encoder = []
	decoder = [] 
	normData = []
	
	if (simType in ['PODG','PODG-MZ','PODG-TCN']): 
		modelLabel = 'param_samp1'
		VMat_full = np.load('./Data/PODBasis/podBasis_'+modelLabel+'.npy')
		VMat = VMat_full[:,:romSize]
		normData = np.load('./Data/PODBasis/normData_'+modelLabel+'.npy')

	elif (simType in ['GMan','GMan-TCN']):
		modelLabel = 'k10_param_kookjin'

		decoderLoc = './Models/decoder_'+modelLabel+'.h5'
		encoderLoc = './Models/encoder_'+modelLabel+'.h5'
		normLoc  = './Data/normData_'+modelLabel+'.npy'

		normData = np.load(normLoc) 
		encoder = load_model(encoderLoc,compile=False)
		decoder = load_model(decoderLoc,compile=False)

		# normLoc = './Data/normData_k10_param_kookjin.npy'
		# modelLoc = './Models/CAE_k10_param_kookjin.h5'
		# # modelLoc = './Models/CAE_k10_param_halfFilters.h5' 

		# numConvLayers = 4
		# encodeFilterList = [8,16,32,64]
		# encodeKernelList = [25,25,25,25]
		# encodeStrideList = [2,4,4,4]
		# encodeActivationList = ['elu','elu','elu','elu']
		# decodeFilterList = [32,16,8,1]
		# decodeKernelList = [25,25,25,25]
		# decodeStrideList = [4,4,4,2]
		# decodeActivationList = ['elu','elu','elu','linear']
		# initDist = 'glorot_uniform'
		# denseActivation = 'elu'
		# decodeDenseInputSize = int(N/np.prod(encodeStrideList))

		# encoder, decoder = extractEncoderDecoder(modelLoc,N,numConvLayers,romSize,
		# 		encodeKernelList,encodeFilterList,encodeStrideList,encodeActivationList,
		# 		decodeKernelList,decodeFilterList,decodeStrideList,decodeActivationList,
		# 		decodeDenseInputSize,denseActivation,initDist)

		# normData = np.load(normLoc)

		# import pdb; pdb.set_trace()

	# MZ parameters 
	mzEps = 1.e-5
	mzTau = 1

	# FOM solution for error calcs 
	if (simType != 'FOM'):
		fomDir = './Data/trainingData'
		
		if (calcProj):
			fomSolLoc = [os.path.join(fomDir,'u_burgers_mu1_'+str(mu_1)+'_mu2_'+str(mu_2)+'_FOM.npy'),
						 os.path.join(fomDir,'RHS_burgers_mu1_'+str(mu_1)+'_mu2_'+str(mu_2)+'_FOM.npy')]
			if (not (os.path.isfile(fomSolLoc[0]) and os.path.isfile(fomSolLoc[1]))): raise ValueError('FOM files not found')
		else:
			if (compareType == 'u'): 
				fomSolLoc = os.path.join(fomDir,'u_burgers_mu1_'+str(mu_1)+'_mu2_'+str(mu_2)+'_FOM.npy')
			elif (compareType == 'RHS'):
				fomSolLoc = os.path.join(fomDir,'RHS_burgers_mu1_'+str(mu_1)+'_mu2_'+str(mu_2)+'_FOM.npy')
			else:
				raise ValueError("Invalid comparison flag")
			if (not os.path.isfile(fomSolLoc)): raise ValueError('FOM file not found at '+fomSolLoc)

		romParams = {'fomSolLoc':fomSolLoc,'VMat':VMat,'romSize':romSize,'u0':u0,
					'encoder':encoder,'decoder':decoder,'normData':normData,
					'mzEps':mzEps,'mzTau':mzTau}
	else:
		romParams = {}

	############# RUNTIME FUNCS #################
	if (not paramBurgers): 
		print("Running single simulation, mu1: "+str(mu_1)+", mu2: "+str(mu_2))

		# create tailored output label for data
		outputLabel = problem+'_mu1_'+str(mu_1)+'_mu2_'+str(mu_2)+'_'+simType
		if not os.path.exists('./Images/snaps_'+outputLabel): os.makedirs('./Images/snaps_'+outputLabel)

		if calcProj:
			computeProjSol(simType,u0,linOp,bc_vec,source_term,x,dt,dx,tEnd,
				spaceDiffParams,romParams,plotSol,sampRate,plotSnaps,outputLabel,
				saveSol,saveRHS,saveCode,calcErr,outputLoc)
		else:
			# run full-order model simulation
			computeSol(simType,u0,linOp,bc_vec,source_term,x,dt,dx,tEnd,Nt,
				spaceDiffParams,timeDiffParams,romParams,plotSol,sampRate,plotSnaps,
				restartFlag,restartFile,outputLabel,saveSol,saveRHS,calcErr,outputLoc,compareType) 

	else:
		print("Running parameterized barrage")
		for paramIter,mu_1 in enumerate(MU_1_vec):
			# recompute source term and spatial differentiation operators given parameter set
			mu_2 = MU_2_vec[paramIter]
			source_term = param_mult*np.exp(mu_2*x)
			spaceDiffParams['bc_vals'][0] = mu_1
			linOp, bc_vec = spaceSchemes.precompLinOp(problem,spaceDiffParams,N,dx,viscosity)

			outputLabel = problem+'_mu1_'+str(mu_1)+'_mu2_'+str(mu_2)+'_'+simType 

			if (simType != 'FOM'):
				if (compareType == 'u'): 
					fomSolLoc = os.path.join(fomDir,'u_burgers_mu1_'+str(mu_1)+'_mu2_'+str(mu_2)+'_FOM.npy')
				elif (compareType == 'RHS'):
					fomSolLoc = os.path.join(fomDir,'RHS_burgers_mu1_'+str(mu_1)+'_mu2_'+str(mu_2)+'_FOM.npy')
				else:
					raise ValueError("Invalid comparison flag")
				if (not os.path.isfile(fomSolLoc)): raise ValueError('FOM file not found at '+fomSolLoc)

			if calcProj: 
				computeProjSol(simType,u0,linOp,bc_vec,source_term,x,dt,dx,tEnd,
					spaceDiffParams,romParams,plotSol,sampRate,plotSnaps,outputLabel,
					saveSol,saveRHS,saveCode,calcErr,outputLoc)
			else: 
				computeSol(simType,u0,linOp,bc_vec,source_term,x,dt,dx,tEnd,Nt,
					spaceDiffParams,timeDiffParams,romParams,plotSol,sampRate,plotSnaps,
					restartFlag,restartFile,outputLabel,saveSol,saveRHS,calcErr,outputLoc,compareType) 

if __name__ == "__main__":
	main()
