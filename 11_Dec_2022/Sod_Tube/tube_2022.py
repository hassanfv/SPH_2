
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit


@njit
def kernel_h(x, h):

	sig = 2./3.
	
	N = len(x)
	W = np.zeros((N, N))

	for i in range(N):
		for j in range(N):
		
			xij = x[i] - x[j]
			rij = np.sqrt(xij**2) # 1D
			
			hij = 0.5 * (h[i] + h[j])
			
			sij = rij / hij
			
			if sij <= 1.0:
				
				W[i][j] = sig / hij * ( (3./2.) * (0.5 * sij - 1.) * sij**2 + 1.)
			
			elif (sij >= 1.) and (sij <= 2.):
				
				W[i][j] = sig / hij * (1./4.) * (2. - sij)**3
				
	return W



@njit
def grad_kernel_h(x, h):

	sig = 2./3.
	
	N = len(x)
	gW = np.zeros((N, N))

	for i in range(N):
		for j in range(N):
		
			xij = x[i] - x[j]
			rij = np.sqrt(xij**2) + 1e-8 # 1D
			
			hij = 0.5 * (h[i] + h[j])
			
			sij = rij / hij
			
			if sij <= 1.0:
				
				gW[i][j] = -sig / hij * xij / hij / rij * ( 3. * sij - 9./4. * sij**2)
			
			elif (sij >= 1.) and (sij <= 2.):
				
				gW[i][j] = -sig / hij * xij / hij / rij * 3. / 4. * (2. - sij)**2
				
	return gW


@njit
def Art_visc(x, v, rho, c, h):

	PI = np.zeros((N, N))

	for i in range(N):
		for j in range(N):

			hij = 0.5 * (h[i] + h[j])
			vij = v[i] - v[j]
			xij = x[i] - x[j]
			rij = np.sqrt(xij**2)
			rhoij = 0.5 * (rho[i] + rho[j])
			cij = 0.5 * (c[i] + c[j])

			numerator = vij * xij
			denominator = rij**2 + hij**2 * eta**2
			muij = hij * numerator / denominator
			
			PI[i][j] = 1./rhoij * (-1. * alpha * muij * cij + beta * muij**2)

	return PI


@njit
def density_h(x, m, h):

	N = len(m)
	rho = np.zeros(N)
	
	W = kernel_h(x, h)
	
	for i in range(N):
		s = 0.0
		for j in range(N):
		
			s += m[j] * W[i][j]
			
		rho[i] = s

	return rho


#x = np.arange(-1., 1., 0.001)
#h = np.zeros_like(x) + 0.10

#*********************************************
#*********** Initial Condition ***************

xLeft = np.linspace(-0.5, 0., 2500)
dxL = xLeft[1] - xLeft[0]
xRight = np.linspace(0, 0.5, 314)
dxR = xRight[1] - xRight[0]

Left_boundary = np.linspace(-0.5 - (50 * dxL), -0.5 - dxL, 50)
Right_boundary = np.linspace(0.5 + dxR, 0.5 + (50 * dxR), 50)

xLeft = np.append(Left_boundary, xLeft)
xRight = np.append(xRight, Right_boundary)

x = np.append(xLeft, xRight)

h = np.zeros_like(x) + 2. * dxR # We take 2*dxR as the smoothing length

#rho1 = np.append(np.ones_like(xLeft), np.ones_like(xRight) * 0.125)

m = np.zeros_like(x) + 0.0002
rho = density_h(x, m, h)
rho[:50] = 1.0
rho[-20:] = 0.125

P = np.append(np.ones_like(xLeft), np.ones_like(xRight) * 0.1)

gamma = 5./3.
u = P / ((gamma - 1.) * rho)

plt.scatter(x, u, s = 1)

#plt.scatter(x, rho1, s = 1, color = 'black')
#plt.scatter(x, P, s = 1, color = 'blue')


plt.show()








