"""
Driver for computing the finite difference approximate solution to the Burgers' or Kuramoto-Sivashinsky equation
	Burgers': u_t = - u*u_x + nu*u_xx 
	Kuramoto-Sivashinsky: u_t = - u*u_x - u_xx - nu*u_xxxx
Author: Christopher Wentland, University of Michigan
Date: August 2019
"""

import math
import os
import numpy as np
import spaceSchemes
from timeSchemes import storeTimeDiffParams
import onlineFuncs
import matplotlib.pyplot as plt
from annFuncs import extractEncoderDecoder

def main():

	if not os.path.exists('./Data'): os.makedirs('./Data')
	if not os.path.exists('./Images'): os.makedirs('./Images')
	if not os.path.exists('./RestartFiles'): os.makedirs('./RestartFiles')

	simType = 'FOM'  # 'FOM', 'PODG', 'PODG-MZ', 'PODG-TCN', 'GMan', 'GMan-TCN'
	paramBurgers = False

	# governing equation selection
	problem = 'burgers'		# 'burgers': Burgers' equation, 'ks': Kuramoto-Sivashinsky equation
	saveSol = True
	saveRHS = False
	plotSol = True 		# plot the FOM solution (can take a bit for fine grids/time sampling, long time evolution)
	plotSnaps = False

	# temporal and spatial derivative approximates
	# note: linear second-order and fourth-order derivatives are always computed by central schemes
	timeDiffScheme = 'RK'				# 'BDF': backwards differentiation (implicit), 'RK': Runge-Kutta (explicit) 
	timeOrdAcc = 4						# 1-4 (first-order to fourth-order accuracy)
	nonlinDiffScheme = 'upwind'		# 'upwind': upwind scheme, 'central': central scheme
	nonlinOrdAcc = 1					# 1,2,4 (first-order to third-order accuracy)
										# note: upwind scheme only accepts 1; central scheme only accepts 2,4
	linOrdAcc = 2 						# 2, 4 (second-order or fourth-order accuracy)

	# boundary conditions
	bound_cond = 'dirichlet'
	bc_vals = [4.3,1]

	spaceDiffParams = {'nonlinDiffScheme':nonlinDiffScheme,'nonlinOrdAcc':nonlinOrdAcc,'linOrdAcc':linOrdAcc,'bound_cond':bound_cond,'bc_vals':bc_vals}
	timeDiffParams = storeTimeDiffParams(timeDiffScheme,timeOrdAcc)

	# spatial domain settings
	N = 256							# number of spatial points for finite-difference discretization
	xi = 0.0						# x-coordinate of 'left' boundary
	xf = 100. 				# x-coordinate of 'right' (periodic) boundary
	dx = (xf-xi)/float(N-1)			# uniform distance between spatial points
	x = np.linspace(xi,xf,N) 	# vector of spatial points

	# set initial conditions
	ICType = 'uniform' 									# 'turbulent': mock-turbulent/Burgulence 
															# 'sin': sine wave
															# 'mixedSinCos': product of cosine and offset sine 
	unif_val = 1.
	randSeed = 7											# fixed seed for RNG in 'turbulence' IC
	numWaves = 12											# number of waves in 'turbulence' IC
	angFreq = 1. 										# angular frequency for 'sin' or mixedSinCos' ICs 
	u0 = spaceSchemes.setICs(ICType,x,unif_val,randSeed,numWaves,angFreq)	# initial condition field

	# temporal integration parameters
	restartFlag = False 					# restart from restart file
	restartFile = 'restart_ks_8000' 			# name of restart npy file, sans file extension
										# note: I just manually create restart files from solution save data
										# 	Just append the iteration number and time to the beginning of the snapshot array and save
	tEnd = 35. 						# end time of simulation
	dt = 0.07 							# time step
	Nt = int(round(tEnd/dt)) 			# total number of time steps from t = 0 to tEnd
	sampRate = 1 						# time step interval for saving solution to disk

	# parameterized Burgers'
	param_mult = 0.02
	mu_2 = 0.021
	source_term = param_mult*np.exp(mu_2*x)

	# compute discrete linear operator matrix
	viscosity = 0.																# dissipation coefficient
	linOp, bc_vec = spaceSchemes.precompLinOp(problem,spaceDiffParams,N,dx,viscosity) 		# compute linear dissipation operator

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
	
	if (simType in ['PODG','PODG-MZ','PODG-TCN']):
		VMat_full = np.load('./Data/PODBasis/podBasis.npy')
		VMat = VMat_full[:,:romSize]


	elif (simType in ['GMan','GMan-TCN']):
		modelLoc = './Models/model_test_save.h5'
		numConvLayers = 4
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

		import pdb; pdb.set_trace()

	# MZ parameters 
	mzEps = 1.e-5
	mzTau = 1

	# error calcs
	fomSolLoc = './Data/u_burgers_mu1_4.3_mu2_0.021_FOM.npy'

	romParams = {'fomSolLoc':fomSolLoc,'VMat':VMat,'romSize':romSize,'encoder':encoder,'decoder':decoder,'mzEps':mzEps,'mzTau':mzTau}



	############# RUNTIME FUNCS #################
	if (not paramBurgers):
		# create tailored output label for data
		outputLabel = problem+'_mu1_'+str(bc_vals[0])+'_mu2_'+str(mu_2)+'_'+simType

		# run full-order model simulation
		onlineFuncs.computeSol(simType,u0,linOp,bc_vec,source_term,x,dt,dx,tEnd,Nt,
			spaceDiffParams,timeDiffParams,romParams,problem,plotSol,sampRate,plotSnaps,restartFlag,restartFile,outputLabel,saveSol,saveRHS) 
	else:

		for paramIter,mu_1 in enumerate(MU_1_vec):
			# recompute source term and spatial differentiation operators given parameter set
			mu_2 = MU_2_vec[paramIter]
			source_term = param_mult*np.exp(mu_2*x)
			spaceDiffParams['bc_vals'][0] = mu_1
			linOp, bc_vec = spaceSchemes.precompLinOp(problem,spaceDiffParams,N,dx,viscosity)

			outputLabel = problem+'_mu1_'+str(mu_1)+'_mu2_'+str(mu_2)+'_'+simType
			onlineFuncs.computeSol(simType,u0,linOp,bc_vec,source_term,x,dt,dx,tEnd,Nt,
				spaceDiffParams,timeDiffParams,romParams,problem,plotSol,sampRate,plotSnaps,restartFlag,restartFile,outputLabel,saveSol,saveRHS) 

if __name__ == "__main__":
	main()