"""
'Online' calculation and storage of approximate solution of Burgers/K-S equation 
Author: Christopher Wentland, University of Michigan
Date: August 2019
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import math
import time
import timeSchemes
from spaceSchemes import computeNonlinRHS

# compute full-order model solution, write solution and RHS function to disk
def computeFOM(u0,linOp,x,dt,dx,tEnd,Nt,spaceDiffParams,timeDiffParams,problem,plotSol,sampRate,restartFlag,restartFile):

	# if initializing from restart file, read in restart solution and iteration
	if restartFlag:	
		restartData = np.load(os.path.join('./RestartFiles',restartFile+'.npy'))
		restartIter = int(restartData[-1])
		u = restartData[:-1]
	# otherwise initialize from initial conditions
	else:
		restartIter = 0
		u = u0.copy()

	restartTime = restartIter*dt

	print(Nt)
	print(restartIter)
	print(sampRate)
	print(math.floor((Nt-restartIter)/sampRate))
	print(u.shape)
	# store time evolution for write to disk and plotting
	uSave = np.zeros((int(math.floor((Nt-restartIter)/sampRate)),u.shape[0]))
	RHSSave = np.zeros((int(math.floor((Nt-restartIter)/sampRate)),u.shape[0]))
	uSave[0,:] = u.copy()
	RHS = computeNonlinRHS(u,dx,spaceDiffParams) + np.dot(linOp,u.T).T
	RHSSave[0,:] = RHS.copy()

	# initialize past state matrix for linear multistep scheme
	if timeDiffParams['timeDiffScheme'] == 'BDF':
		uMat = np.zeros((timeDiffParams['timeOrdAcc']+1,u.shape[0]),dtype=np.float64)
		uMat[0,:] = u.copy()
		uMat[1,:] = u.copy()


	startTime = time.time()
	counter = 1
	for i in range(restartIter,Nt,1):

		print('FOM outer iteration '+str(i+1))

		if timeDiffParams['timeDiffScheme'] == 'RK':
			u, RHS = timeSchemes.advanceTimeStepExplicitRungeKutta(u,RHS,linOp,dx,dt,spaceDiffParams,timeDiffParams)
		elif timeDiffParams['timeDiffScheme'] == 'BDF':
			# for 'cold start', must use lowest accuracy scheme as we don't have access to u_n-1, u_n-2, etc.
			schemeSwitch = min(counter,timeDiffParams['timeOrdAcc'])
			timeDiffParams['BDFTimeDerivCoeff'] = timeDiffParams['uNp1Coeffs'][schemeSwitch-1]
			uMat, RHS = timeSchemes.advanceTimeStepBDF(uMat,timeDiffParams,spaceDiffParams,dt,dx,linOp,schemeSwitch)
			u = uMat[0,:]
			counter += 1
		else:
			raise ValueError('Invalid time difference scheme')		

		# store solution and RHS function
		if ((i % sampRate) == 0):
			uSave[int((i-restartIter)/sampRate),:] = u
			RHSSave[int((i-restartIter)/sampRate),:] = RHS

	endTime = time.time()
	print('FOM wall clock time: ' + str(endTime - startTime) + ' seconds')

	# plot filled contour plot of time evolution of solution
	if plotSol:
		t = np.linspace(restartTime,tEnd,math.floor((Nt-restartIter)/sampRate))
		X, T = np.meshgrid(x,t)
		figContourPlot = plt.figure()
		axContourPlot = figContourPlot.add_subplot(111)
		axContourPlot.contourf(X,T,uSave)
		plt.savefig('./Images/contour_'+problem+'_'+str(restartTime)+'_to_'+str(tEnd)+'_FOM.png')

	# save solution and RHS function to disk
	np.save(os.path.join('./Data','u_'+problem+'_'+str(restartTime)+'_to_'+str(tEnd)+'_FOM.npy'),uSave)
	np.save(os.path.join('./Data','RHS_'+problem+'_'+str(restartTime)+'_to_'+str(tEnd)+'_FOM.npy'),RHSSave)