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
from annFuncs import evalEncoder, evalDecoder, evalCAE, extractJacobian, invScaleOp
from podFuncs import evalPODProj, evalPODLift, scaleOp, evalPODFullProj


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
		spaceDiffParams,timeDiffParams,romParams,plotSol,sampRate,plotSnaps,restartFlag,restartFile,outputLabel,
		saveSol,saveRHS,calcErr,outputLoc,compareType):

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

	if (simType in ['PODG','PODG-MZ','PODG-TCN']):
		a = evalPODProj(u, u0, romParams['VMat'],romParams['normData'])
		u = evalPODLift(a,u0,romParams['VMat'],romParams['normData']) 
		RHS = computeNonlinRHS(u,dx,spaceDiffParams) + np.dot(linOp,u.T).T + bc_vec + source_term
		jacob = np.empty(1)
	elif (simType in ['GMan','GMan-TCN']): 
		encoder = romParams['encoder'] 
		decoder = romParams['decoder']
		a = evalEncoder(u,u0,encoder,romParams['normData']) 
		# jacob = extractJacobian(decoder,a)
		# u = invScaleOp(u, romParams['normData']) + u0 
		u = evalDecoder(a,u0,decoder,romParams['normData'])
		RHS = computeNonlinRHS(u,dx,spaceDiffParams) + np.dot(linOp,u.T).T + bc_vec + source_term
	elif (simType == 'FOM'):
		a = np.empty(1) 
		jacob = np.empty(1)


	startTime = time.time()
	counter = 1
	for i in range(restartIter,Nt,1):

		print(simType+' outer iteration '+str(i+1))

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
		if (compareType == 'u'):
			dataPlot = uSave
		elif (compareType == 'RHS'):
			dataPlot = RHSSave 

		if calcErr: calcPlotErr(fomSol,dataPlot,t,outputLabel)

		if (plotSnaps):

			figLinePlot = plt.figure()
			axLinePlot  = figLinePlot.add_subplot(111)
			dataMax = max(np.amax(fomSol),np.amax(dataPlot))
			dataMin = min(np.amin(fomSol),np.amin(dataPlot)) 
			for t in range(restartIter,Nt,sampRate):
				axLinePlot.cla()
				axLinePlot.plot(x,fomSol[:,t],'k')
				axLinePlot.plot(x,dataPlot[t,:],'r')
				axLinePlot.set_ylim([dataMin,dataMax]) 
				plt.pause(0.01)

	# save solution and RHS function to disk
	if saveSol: np.save(os.path.join(outputLoc,'u_'+outputLabel+'.npy'),uSave.T)
	if saveRHS: np.save(os.path.join(outputLoc,'RHS_'+outputLabel+'.npy'),RHSSave.T)

def computeProjSol(simType,u0,linOp,bc_vec,source_term,x,dt,dx,tEnd,
		spaceDiffParams,romParams,plotSol,sampRate,plotSnaps,outputLabel,
		saveSol,saveRHS,saveCode,calcErr,outputLoc):

	fomSol_u = np.load(romParams['fomSolLoc'][0])
	fomSol_RHS = np.load(romParams['fomSolLoc'][1])
	_, numSamps = fomSol_u.shape

	# store time evolution for write to disk and plotting
	uSave = np.zeros(fomSol_u.shape,dtype=np.float64).T
	RHSSave = np.zeros(fomSol_u.shape,dtype=np.float64).T
	codeSave = np.zeros((romParams['romSize'],numSamps),dtype=np.float64).T
	closureSave = np.zeros((romParams['romSize'],numSamps),dtype=np.float64).T

	# figRHS = plt.figure()
	# axRHS = figRHS.add_subplot(111)
	# axRHS2 = axRHS.twinx()
	modeLin = np.linspace(1,romParams['romSize'],romParams['romSize'])

	if (simType == 'PODG'):
		VMat = romParams['VMat']
	elif (simType == 'GMan'):
		encoder = romParams['encoder']
		decoder = romParams['decoder']

	startTime = time.time()
	counter = 1
	for i in range(numSamps):
		
		print(simType+' projection iteration '+str(i+1))

		fomSnap_u = fomSol_u[:,i]
		fomSnap_RHS = fomSol_RHS[:,i]

		if (simType == 'PODG'): 
			code = evalPODProj(np.squeeze(fomSnap_u),u0,VMat,romParams['normData'])
			u = evalPODLift(np.squeeze(code),u0,VMat,romParams['normData'])
		elif (simType == 'GMan'):
			code = evalEncoder(np.squeeze(fomSnap_u),u0,encoder,romParams['normData'])
			u, jacob = extractJacobian(decoder,code) 
			u = invScaleOp(u, romParams['normData']) + u0
		else:
			raise ValueError("Invalid projection, must use PODG or GMan")

		nonlin = computeNonlinRHS(u,dx,spaceDiffParams) 
		lin = np.dot(linOp,u.T).T
		RHS = nonlin + lin + bc_vec + source_term 

		##################### THIS NEEDS TO BE VERIFIED ############
		RHS[0] = RHS[1] # getting rid of HUGE flux from boundary condition 
		############################################################

		if (simType == 'PODG'):
			closure = np.dot(VMat.T,(fomSnap_RHS - RHS).T).T 
			closure = scaleOp(closure,romParams['normData'])
		elif (simType == 'GMan'):
			closure = np.dot(np.linalg.pinv(jacob),(fomSnap_RHS - RHS).T).T 
			closure = scaleOp(closure,romParams['normData'])

		# axRHS.plot(x,nonlin)
		# axRHS.plot(x,lin)
		# axRHS.plot(x,bc_vec)
		# axRHS.plot(x,source_term)
		# axRHS.legend(['nonlinear','linear','BC vec','source'])
		# axRHS.plot(modeLin,code,'b')
		# axRHS2.plot(modeLin,closure,'r')
		# plt.pause(0.01)
		# axRHS.clear()
		# axRHS2.clear()

		# store solution and RHS function
		if ((i % sampRate) == 0):
			sampNum = int(i/sampRate)
			uSave[sampNum,:] = u
			RHSSave[sampNum,:] = RHS 
			codeSave[sampNum,:] = code
			closureSave[sampNum,:] = closure

	endTime = time.time()
	print(simType+' wall clock time: ' + str(endTime - startTime) + ' seconds')

	# plot filled contour plot of time evolution of solution
	t = np.linspace(0,tEnd,math.floor(numSamps/sampRate))
	if plotSol:
		X, T = np.meshgrid(x,t)
		figContourPlot = plt.figure()
		axContourPlot = figContourPlot.add_subplot(111)
		ctr = axContourPlot.contourf(X,T,uSave)
		axContourPlot.set_xlabel('x')
		axContourPlot.set_ylabel('t')
		plt.colorbar(ctr,ax=axContourPlot)
		plt.savefig('./Images/contour_'+outputLabel+'.png')
		
	if calcErr: 
		
		calcPlotErr(fomSol_u,uSave,t,outputLabel)

	if (plotSnaps):
		figLinePlot, axLinePlot = plt.subplots(nrows=3) 
		figLinePlot.set_size_inches(10,10)
		axClosure = axLinePlot[-1].twinx()
		dataMax_u = max(np.amax(fomSol_u),np.amax(uSave))
		dataMin_u = min(np.amin(fomSol_u),np.amin(uSave))
		dataMax_RHS = max(np.amax(fomSol_RHS),np.amax(RHSSave))
		dataMin_RHS = min(np.amin(fomSol_RHS),np.amin(RHSSave)) 
		# dataMax_code = np.amax(codeSave); dataMin_code = np.amin(codeSave)
		# dataMax_closure = np.amax(closureSave); dataMin_closure = np.amin(closureSave)
		dataBound_code = max(abs(np.amax(codeSave)), abs(np.amin(codeSave)))
		dataBound_closure = max(abs(np.amax(closureSave)), abs(np.amin(closureSave)))
		for t in range(0,numSamps,sampRate):
			axLinePlot[0].clear()
			axLinePlot[1].clear()
			axLinePlot[2].clear()
			axClosure.clear()

			axLinePlot[0].plot(x,fomSol_u[:,t],'k')
			axLinePlot[0].plot(x,uSave[t,:],'r')
			axLinePlot[0].set_ylim([dataMin_u,dataMax_u])

			axLinePlot[1].plot(x,fomSol_RHS[:,t],'k')
			axLinePlot[1].plot(x,RHSSave[t,:],'r')
			axLinePlot[1].set_ylim([dataMin_RHS,dataMax_RHS])

			axLinePlot[2].bar(modeLin,codeSave[t,:],color='k',width=1.0)
			# axLinePlot[2].set_ylim([dataMin_code,dataMax_code])
			axLinePlot[2].set_ylim([-dataBound_code,dataBound_code])
			axClosure.bar(modeLin,closureSave[t,:],color='r',width=0.5)
			# axClosure.set_ylim([dataMin_closure,dataMax_closure])
			axClosure.set_ylim([-dataBound_closure,dataBound_closure])

			plt.pause(0.01)

	# save solution and RHS function to disk
	if saveSol: np.save(os.path.join(outputLoc,'u_'+outputLabel+'.npy'),uSave.T)
	if saveRHS: np.save(os.path.join(outputLoc,'RHS_'+outputLabel+'.npy'),RHSSave.T) 
	if saveCode: np.save(os.path.join(outputLoc,'code_'+outputLabel+'.npy'),codeSave.T)