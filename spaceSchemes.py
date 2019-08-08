"""
Spatial finite difference scheme functions for computing approximate solution of Burgers'/K-S equation
Author: Christopher Wentland, University of Michigan
Date: August 2019
"""

import numpy as np
from numpy.random import seed,rand
from numpy.fft import fftshift
from math import pi, pow, sqrt
from scipy.sparse import diags, spdiags

# precompute linear operator for second/fourth-order derivative discretizations 
def precompLinOp(problem,spaceDiffParams,N,dx,viscosity):

	# set up second derivative linear operator
	if spaceDiffParams['linOrdAcc'] == 2:
		viscDiag = np.ones((3,N-1),dtype=np.float64)
		viscDiag[1,:] *= -2.
		secondDerivOp = spdiags(viscDiag,np.array([-1,0,1]),N-1,N-1).toarray()
		secondDerivOp[0,-1] = 1.
		secondDerivOp[-1,0] = 1.
	elif spaceDiffParams['linOrdAcc'] == 4: 
		viscDiag = np.ones((5,N-1),dtype=np.float64)
		viscDiag[0,:] *= -1./12.
		viscDiag[1,:] *= 4./3.
		viscDiag[2,:] *= -5./2.
		viscDiag[3,:] *= 4./3.
		viscDiag[4,:] *= -1./12.
		secondDerivOp = spdiags(viscDiag,np.array([-2,-1,0,1,2]),N-1,N-1).toarray()
		secondDerivOp[0,-1] = 4./3.
		secondDerivOp[0,-2] = -1./12.
		secondDerivOp[1,-1] = -1./12.
		secondDerivOp[-1,0] = 4./3.
		secondDerivOp[-2,0] = -1./12.
		secondDerivOp[-1,1] = -1./12.
	else:
		raise ValueError('Invalid linear operator order of accuracy input')

	# set up fourth derivative linear operator
	if problem == 'ks':
		if spaceDiffParams['linOrdAcc'] == 2:
			viscDiag = np.ones((5,N-1),dtype=np.float64)
			viscDiag[0,:] *= 1.
			viscDiag[1,:] *= -4.
			viscDiag[2,:] *= 6.
			viscDiag[3,:] *= -4.
			viscDiag[4,:] *= 1.
			fourthDerivOp = spdiags(viscDiag,np.array([-2,-1,0,1,2]),N-1,N-1).toarray()
			fourthDerivOp[0,-1] = -4.
			fourthDerivOp[0,-2] = 1.
			fourthDerivOp[1,-1] = 1.
			fourthDerivOp[-1,0] = -4.
			fourthDerivOp[-1,1] = 1.
			fourthDerivOp[-2,0] = 1.

		elif spaceDiffParams['linOrdAcc'] == 4:
			viscDiag = np.ones((7,N-1),dtype=np.float64)
			viscDiag[0,:] *= -1./6.
			viscDiag[1,:] *= 2.
			viscDiag[2,:] *= -13./2.
			viscDiag[3,:] *= 28./3.
			viscDiag[4,:] *= -13./2.
			viscDiag[5,:] *= 2.
			viscDiag[6,:] *= -1./6.
			fourthDerivOp = spdiags(viscDiag,np.array([-3,-2,-1,0,1,2,3]),N-1,N-1).toarray()
			fourthDerivOp[0,-1] = -13./2.
			fourthDerivOp[0,-2] = 2.
			fourthDerivOp[1,-1] = 2.
			fourthDerivOp[0,-3] = -1./6.
			fourthDerivOp[1,-2] = -1./6.
			fourthDerivOp[2,-1] = -1./6.
			fourthDerivOp[-1,0] = -13./2.
			fourthDerivOp[-1,1] = 2.
			fourthDerivOp[-2,0] = 2.
			fourthDerivOp[-1,2] = -1./6.
			fourthDerivOp[-2,1] = -1./6.
			fourthDerivOp[-3,0] = -1./6.

		else:
			raise ValueError('Invalid linear operator order of accuracy input')
		
		# linear operator for K-S equation, i.e. -u_xx - nu*u_xxxx
		linOp = -secondDerivOp/dx**2 - fourthDerivOp*viscosity/dx**4

	elif (problem == 'burgers'):
		# linear operator for Burgers' equation, i.e. u_xx
		linOp = secondDerivOp*viscosity/dx**2
	
	else:
		raise ValueError('Invalid problem choice')

	return linOp

# set initial conditions
def setICs(ICType,x,rand_seed,num_waves,angFreq):
	nx = x.shape[0]
	u0 = np.zeros((nx),dtype=np.float64)
	
	# mock turbulent initial conditions, random superposition of waves satisfying -5/3 energy decay law
	if (ICType == 'turbulent'):
		E = np.zeros((num_waves),dtype=np.float64)
		seed(rand_seed)
		beta = fftshift(rand(nx)*2.*pi - pi)

		for i in range(0,5,1):
			E[i] = pow(5.,-5./3.)
			u0 += sqrt(2.*E[i])*np.sin(float(i)*x + beta[i]);

		for i in range(5,num_waves,1):
			E[i] = pow(float(i),-5./3.)
			u0[:] += sqrt(2.*E[i])*np.sin(float(i)*x + beta[i])

	# simple sine wave
	elif (ICType == 'sin'):
		u0 = np.sin(x*angFreq)

	# product of cosine and offset sine wave
	elif (ICType == 'mixedSinCos'):
		u0 = np.cos(x*angFreq)*(1.+np.sin(x*angFreq))

	else:
		raise ValueError('Invalid initial condition selection')
	
	return u0

# compute nonlinear component of RHS function
def computeNonlinRHS(u,dx,spaceDiffParams):

	# central difference scheme
	if spaceDiffParams['nonlinDiffScheme'] == 'central':
		if spaceDiffParams['nonlinOrdAcc'] == 2:
			RHS =  0.5*u*(np.roll(u,1) - np.roll(u,-1))
		elif spaceDiffParams['nonlinOrdAcc'] == 4:
			RHS = u*(1./12.*(np.roll(u,-2) - np.roll(u,2)) + 2./3.*(np.roll(u,1) - np.roll(u,-1)))
		else:
			raise ValueError('Invalid nonlinear operator order of accuracy')

	# upwind scheme
	elif spaceDiffParams['nonlinDiffScheme'] == 'upwind':
		if spaceDiffParams['nonlinOrdAcc'] == 1:
			R = (u+np.absolute(u))*u/4.
			R_plus = R - np.roll(R,1)
			R = (u-np.absolute(u))*u/4.
			R_minus = np.roll(R,-1) - R
			RHS = -(R_plus+R_minus).T
		else:
			raise ValueError('Invalid nonlinear operator order of accuracy')

	return RHS/dx

# calculate analytical Jacobian of RHS function, i.e. J = dR/du
# These are exact formulations of the Jacobian, no numerical approximation is made
def computeAnalyticalRHSJacobian(u,linOp,dx,spaceDiffParams):

	# nonlinear RHS Jacobian, central difference scheme
	if spaceDiffParams['nonlinDiffScheme'] == 'central':
		if spaceDiffParams['nonlinOrdAcc'] == 2:
			u_j = u/(2.*dx)
			u_j_p1 = np.roll(u_j,-1)
			u_j_m1 = np.roll(u_j,1)

			diag_0 = (u_j_m1 - u_j_p1)
			diag_m1 = u_j
			diag_p1 = -u_j

			diagonals = [diag_m1[1:], diag_0, diag_p1[:-1]]

			nonlinJacob = diags(diagonals,[-1,0,1],dtype=np.float64).toarray()
			nonlinJacob[0,-1] = diag_m1[0]
			nonlinJacob[-1,0] = diag_p1[-1]

		elif spaceDiffParams['nonlinOrdAcc'] == 4:
			u_j = u/dx 
			u_j_m1 = np.roll(u_j,1)
			u_j_m2 = np.roll(u_j,2)
			u_j_p1 = np.roll(u_j,-1)
			u_j_p2 = np.roll(u_j,-2)

			diag_0 = (1./12.)*u_j_p2 - (2./3.)*u_j_p1 + (2./3.)*u_j_m1 - (1./12.)*u_j_m2
			diag_p2 = (1./12.)*u_j 
			diag_p1 = (-2./3.)*u_j 
			diag_m1 = (2./3.)*u_j 
			diag_m2 = (-1./12.)*u_j 

			diagonals = [diag_m2[2:], diag_m1[1:], diag_0, diag_p1[:-1], diag_p2[:-2]]
			nonlinJacob = diags(diagonals,[-2,-1,0,1,2],dtype=np.float64).toarray()
			nonlinJacob[0,-1] = diag_m1[0]
			nonlinJacob[0,-2] = diag_m2[0]
			nonlinJacob[1,-1] = diag_m2[1]
			nonlinJacob[-1,0] = diag_p1[-1]
			nonlinJacob[-1,1] = diag_p2[-1]
			nonlinJacob[-2,0] = diag_p2[-2] 

	# nonlinear RHS Jacobian, upwind scheme
	elif spaceDiffParams['nonlinDiffScheme'] == 'upwind':

		u_j = u.copy()
		u_j_p1 = np.roll(u_j,-1)
		u_j_m1 = np.roll(u_j,1)

		u_j_abs = np.absolute(u_j)
		u_j_p1_abs = np.roll(u_j_abs,-1)
		u_j_m1_abs = np.roll(u_j_abs,1)

		u_j_sgn = np.sign(u_j)
		u_j_p1_sgn = np.roll(u_j_sgn,-1)
		u_j_m1_sgn = np.roll(u_j_sgn,1)

		diag_0 = (u_j_abs + u_j*u_j_sgn)/(2.*dx)
		diag_p1 = (u_j_p1*(2. - u_j_p1_sgn) - u_j_p1_abs)/(4.*dx)
		diag_m1 = (u_j_m1*(-2. - u_j_m1_sgn) - u_j_m1_abs)/(4.*dx)

		diagonals = [diag_m1[1:], diag_0, diag_p1[:-1]]

		nonlinJacob = diags(diagonals,[-1,0,1],dtype=np.float64).toarray()
		nonlinJacob[0,-1] = diag_m1[0]
		nonlinJacob[-1,0] = diag_p1[-1]

	else:
		raise ValueError('Invalid nonlinear spatial difference scheme')

	# add linear component contribution	
	J = nonlinJacob + linOp

	return J