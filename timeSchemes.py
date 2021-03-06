"""
Time-stepping schemes for computing approximate solution of Burgers'/K-S equation
Author: Christopher Wentland, University of Michigan
Date: August 2019
"""

import numpy as np
from spaceSchemes import computeNonlinRHS, computeAnalyticalRHSJacobian
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity, csc_matrix

# compute residual of fully-discrete linear system with BDF temporal scheme, given guess of next time step 
# e.g., BDF2 --> r = 1.5*u_n+1 - 2*u_n + 0.5*u_n-1 - dt*R(u_n+1)
def computeResidualBDF(uMat,linOp,dx,dt,spaceDiffParams,schemeSwitch):

	RHS = computeNonlinRHS(uMat[0,:],dx,spaceDiffParams) + np.dot(linOp,uMat[0,:].T).T

	if schemeSwitch == 1:
		timeDiffOp = uMat[0,:]         - uMat[1,:]
	elif schemeSwitch == 2:
		timeDiffOp = 1.5*uMat[0,:]     - 2.*uMat[1,:] + 0.5*uMat[2,:]
	elif schemeSwitch == 3:
		timeDiffOp = 11./6.*uMat[0,:]  - 3.*uMat[1,:] + 1.5*uMat[2,:] - 1./3.*uMat[3,:]
	elif schemeSwitch == 4:
		timeDiffOp = 25./12.*uMat[0,:] - 4.*uMat[1,:] + 3.0*uMat[2,:] - 4./3.*uMat[3,:] + 0.25*uMat[4,:]
	else:
		raise ValueError('Scheme switch failed')

	residual = timeDiffOp - dt*RHS

	return residual, RHS

# compute implicit solution to BDF temporal scheme using Newton's method
# Newton's method: for linear system r(u) = 0, solve [dr(u_i)/du]*du = -r(u_i)
# 	Update solution, u_i+1 = u_i + du
# 	Iterate to converged r =~ 0, or max iteration reached
def advanceTimeStepBDF(uMat,timeDiffParams,spaceDiffParams,dt,dx,linOp,schemeSwitch):

	timeDiffOpDeriv = csc_matrix(timeDiffParams['BDFTimeDerivCoeff']*identity(uMat.shape[1],dtype=np.float64))
	residual, RHS = computeResidualBDF(uMat,linOp,dx,dt,spaceDiffParams,schemeSwitch)
	residualNorm = 1.e2

	iteration = 0
	while ((residualNorm > timeDiffParams['residualTolerance']) and 
			(iteration < timeDiffParams['maxIterations'])):

		iteration += 1

		RHSJacob = computeAnalyticalRHSJacobian(uMat[0,:],linOp,dx,spaceDiffParams)
		residualJacob = timeDiffOpDeriv - dt*csc_matrix(RHSJacob)

		du = spsolve(residualJacob,-residual)
		uMat[0,:] += du

		residual, RHS = computeResidualBDF(uMat,linOp,dx,dt,spaceDiffParams,schemeSwitch)
		residualNorm = np.linalg.norm(residual,ord=2)
		print('BDF inner iteration '+str(iteration)+': ' + str(residualNorm))

	uMat[1:,:] = uMat[0:-1,:]	

	return uMat, RHS

# compute explicit solution to Jameson's low-memory Runge-Kutta temporal scheme
def advanceTimeStepExplicitRungeKutta(u,RHS,linOp,dx,dt,spaceDiffParams,timeDiffParams):

	un = u.copy()
	rkCoeffs = timeDiffParams['rkCoeffs']
	for rk in range(0,rkCoeffs.shape[0],1):
		u = un + dt*rkCoeffs[rk]*RHS
		RHS_nonlin = computeNonlinRHS(u,dx,spaceDiffParams)
		RHS = RHS_nonlin + np.dot(linOp,u.T).T

	return u, RHS

# organize parameters for temporal schemes
def storeTimeDiffParams(timeDiffScheme,timeOrdAcc):

	if timeOrdAcc > 4: raise ValueError('Invalid time scheme order of accuracy')

	# BDF scheme parameters
	if timeDiffScheme == 'BDF':
		maxIterations = 50						# maximum iterations until termination of iterative solution
		residualTolerance = 1.e-10				# residual convergence threshold
		uNp1Coeffs = [1., 1.5, 11./6., 25./12.]	# coefficients of u_n+1 term in BDF residuals
		timeDiffParams = {'maxIterations':maxIterations,'residualTolerance':residualTolerance,
						  'uNp1Coeffs':uNp1Coeffs}

	elif timeDiffScheme == 'RK':
		coeffs = np.array([0.25, 1.0/3.0, 0.5, 1.0],dtype=np.float64) 	# intermediate time stepping coefficients
		rkCoeffs = coeffs[-timeOrdAcc:]
		timeDiffParams = {'rkCoeffs':rkCoeffs}

	else:
		raise ValueError('Invalid time difference scheme')

	timeDiffParams['timeDiffScheme'] = timeDiffScheme		# time stepping scheme name
	timeDiffParams['timeOrdAcc'] = timeOrdAcc				# time stepping order of accuracy

	return timeDiffParams
