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

def main():

	if not os.path.exists('./Data'): os.makedirs('./Data')
	if not os.path.exists('./Images'): os.makedirs('./Images')
	if not os.path.exists('./RestartFiles'): os.makedirs('./RestartFiles')

	# governing equation selection
	problem = 'burgers'		# 'burgers': Burgers' equation, 'ks': Kuramoto-Sivashinsky equation
	plotSol = True 		# plot the FOM solution (can take a bit for fine grids/time sampling, long time evolution)

	# temporal and spatial derivative approximates
	# note: linear second-order and fourth-order derivatives are always computed by central schemes
	timeDiffScheme = 'RK'				# 'BDF': backwards differentiation (implicit), 'RK': Runge-Kutta (explicit) 
	timeOrdAcc = 4						# 1-4 (first-order to fourth-order accuracy)
	nonlinDiffScheme = 'upwind'		# 'upwind': upwind scheme, 'central': central scheme
	nonlinOrdAcc = 1					# 1,2,4 (first-order to third-order accuracy)
										# note: upwind scheme only accepts 1; central scheme only accepts 2,4
	linOrdAcc = 2 						# 2, 4 (second-order or fourth-order accuracy)
	spaceDiffParams = {'nonlinDiffScheme':nonlinDiffScheme,'nonlinOrdAcc':nonlinOrdAcc,'linOrdAcc':linOrdAcc}

	# spatial domain settings
	N = 512							# number of spatial points for finite-difference discretization
	xi = 0.0						# x-coordinate of 'left' boundary
	xf = 2.*math.pi 				# x-coordinate of 'right' (periodic) boundary
	dx = (xf-xi)/float(N-1)			# uniform distance between spatial points
	x = np.linspace(xi,xf-dx,N-1) 	# vector of spatial points

	# set initial conditions
	ICType = 'sin' 									# 'turbulent': mock-turbulent/Burgulence 
															# 'sin': sine wave
															# 'mixedSinCos': product of cosine and offset sine 
	randSeed = 7											# fixed seed for RNG in 'turbulence' IC
	numWaves = 12											# number of waves in 'turbulence' IC
	angFreq = 1. 										# angular frequency for 'sin' or mixedSinCos' ICs 
	u0 = spaceSchemes.setICs(ICType,x,randSeed,numWaves,angFreq)	# initial condition field

	# temporal integration parameters
	restartFlag = False 					# restart from restart file
	restartFile = 'restart_ks_8000' 			# name of restart npy file, sans file extension
										# note: I just manually create restart files from solution save data
										# 	Just append the iteration number and time to the beginning of the snapshot array and save
	tEnd = 2. 						# end time of simulation
	dt = 1.e-3 							# time step
	Nt = int(math.floor(tEnd/dt)) 			# total number of time steps from t = 0 to tEnd
	sampRate = 10 						# time step interval for saving solution to disk
	timeDiffParams = storeTimeDiffParams(timeDiffScheme,timeOrdAcc)

	# compute discrete linear operator matrix
	viscosity = 5.e-3																# dissipation coefficient
	linOp = spaceSchemes.precompLinOp(problem,spaceDiffParams,N,dx,viscosity) 		# compute linear dissipation operator

	# run full-order model simulation
	onlineFuncs.computeFOM(u0,linOp,x,dt,dx,tEnd,Nt,spaceDiffParams,timeDiffParams,problem,plotSol,sampRate,restartFlag,restartFile) 


if __name__ == "__main__":
	main()