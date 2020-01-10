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


def calcPlotErr(fomSol,romSol,t,outputLabel):

	l2Err_time = np.sqrt(np.sum(np.square(fomSol - romSol.T),axis=0))
	l2Err_tot = np.sum(l2Err_time)
	print('Error for sim '+outputLabel+': '+str(l2Err_tot))
	figErr = plt.figure()
	axErr = figErr.add_subplot(111)
	axErr.plot(t,l2Err_time)
	axErr.set_xlabel('t')
	axErr.set_ylabel('L2 Error')
	plt.savefig('./Images/l2Err_'+outputLabel+'.png')


# compute full-order model solution, write solution and RHS function to disk
def computeSol(simType,u0,linOp,bc_vec,source_term,x,dt,dx,tEnd,Nt,
		spaceDiffParams,timeDiffParams,romParams,problem,plotSol,sampRate,plotSnaps,restartFlag,restartFile,outputLabel,saveSol,saveRHS):

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

	# store time evolution for write to disk and plotting
	uSave = np.zeros((int(math.floor((Nt-restartIter)/sampRate)),u.shape[0]))
	RHSSave = np.zeros((int(math.floor((Nt-restartIter)/sampRate)),u.shape[0]))
	uSave[0,:] = u.copy()
	RHS = computeNonlinRHS(u,dx,spaceDiffParams) + np.dot(linOp,u.T).T + bc_vec + source_term
	RHSSave[0,:] = RHS.copy()

	# initialize past state matrix for linear multistep scheme
	if timeDiffParams['timeDiffScheme'] == 'BDF':
		uMat = np.zeros((timeDiffParams['timeOrdAcc']+1,u.shape[0]),dtype=np.float64)
		uMat[0,:] = u.copy()
		uMat[1,:] = u.copy()

	if plotSnaps: 
		figSnaps = plt.figure()
		axSnaps = figSnaps.add_subplot(111)
		if not os.path.exists('./Images/snaps_'+outputLabel): os.makedirs('./Images/snaps_'+outputLabel)

	if (simType in ['PODG','PODG-MZ','PODG-TCN']):
		a = np.dot(romParams['VMat'].T,u.T).T
	elif (simType in ['GMan','GMan-TCN']):
		encoder = romParams['encoder']
		
		import pdb; pdb.set_trace()
	elif (simType == 'FOM'):
		a = np.empty(1)


	startTime = time.time()
	counter = 1
	for i in range(restartIter,Nt,1):

		#print(simType+' outer iteration '+str(i+1))

		if timeDiffParams['timeDiffScheme'] == 'RK':
			u, a, RHS = timeSchemes.advanceTimeStepExplicitRungeKutta(simType,u,a,RHS,linOp,bc_vec,source_term,dx,dt,spaceDiffParams,timeDiffParams,romParams)
		elif timeDiffParams['timeDiffScheme'] == 'BDF':
			# for 'cold start', must use lowest accuracy scheme as we don't have access to u_n-1, u_n-2, etc.
			schemeSwitch = min(counter,timeDiffParams['timeOrdAcc'])
			timeDiffParams['BDFTimeDerivCoeff'] = timeDiffParams['uNp1Coeffs'][schemeSwitch-1]
			uMat, RHS = timeSchemes.advanceTimeStepBDF(uMat,timeDiffParams,spaceDiffParams,dt,dx,linOp,bc_vec,source_term,schemeSwitch)
			u = uMat[0,:]
			counter += 1
		else:
			raise ValueError('Invalid time difference scheme')		

		# store solution and RHS function
		if ((i % sampRate) == 0):
			sampNum = int((i-restartIter)/sampRate)
			uSave[sampNum,:] = u
			RHSSave[sampNum,:] = RHS
			if plotSnaps:
				axSnaps.plot(x,u)
				axSnaps.set_ylim([0.5,6.5])
				axSnaps.set_xlabel('x')
				axSnaps.set_ylabel('u')
				plt.savefig('./Images/snaps_'+outputLabel+'/fig_'+str(sampNum+1)+'.png')
				axSnaps.clear()

	endTime = time.time()
	print(simType+' wall clock time: ' + str(endTime - startTime) + ' seconds')

	# plot filled contour plot of time evolution of solution
	t = np.linspace(restartTime,tEnd,math.floor((Nt-restartIter)/sampRate))
	if plotSol:
		X, T = np.meshgrid(x,t)
		figContourPlot = plt.figure()
		axContourPlot = figContourPlot.add_subplot(111)
		ctr = axContourPlot.contourf(X,T,uSave)
		axContourPlot.set_xlabel('x')
		axContourPlot.set_ylabel('t')
		plt.colorbar(ctr,ax=axContourPlot)
		plt.savefig('./Images/contour_'+outputLabel+'.png')

	if (simType != 'FOM'):
		fomSol = np.load(romParams['fomSolLoc'])
		calcPlotErr(fomSol,uSave,t,outputLabel)

	# save solution and RHS function to disk
	if saveSol: np.save(os.path.join('./Data','u_'+outputLabel+'.npy'),uSave.T)
	if saveRHS: np.save(os.path.join('./Data','RHS_'+outputLabel+'.npy'),RHSSave.T)