
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit
import readchar
import time


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

	N = len(h)
	PI = np.zeros((N, N))
	eta = 0.01

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
			
			if vij * xij >= 0.0:
				muij = 0.0
			
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


@njit
def Acc_Monaghan_1st(x, rho, P, h, m, q_von_bulk):

	N = len(m)
	acc = np.zeros(N)
	gW = grad_kernel_h(x, h)

	for i in range(N):
	
		s = 0.0
		
		for j in range(N):
		
			s += m[j] * ((P[i]+q_von_bulk[i])/rho[i]**2 + (P[j]+q_von_bulk[j])/rho[j]**2) * gW[i][j]
			
		acc[i] = -1.0 * s
	
	return acc



@njit
def U_h_Monaghan_1st(x, v, rho, P, h, q_von_bulk):   # Modify this ~~~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

	N = len(m)
	U = np.zeros(N)
	gW = grad_kernel_h(x, h)
	
	for i in range(N):
	
		s = 0.0
	
		for j in range(N):
		
			vij = v[i] - v[j]
			s += 0.5 * m[j] * ((P[i]+q_von_bulk[i])/rho[i]**2 + (P[j]+q_von_bulk[j])/rho[j]**2) * vij * gW[i][j]
		
		U[i] = s
	
	return U




#def AVisc_Monaghan_First()


@njit
def divVel(x, v, h, rho, m):

	N = len(m)
	divV = np.zeros(N)
	gW = grad_kernel_h(x, h)

	for i in range(N):
		
		s = 0.0
		
		for j in range(N):
		
			vji = v[j] - v[i]
			
			s += m[j] / rho[i] * vji * gW[i][j]

		divV[i] = s

	return divV



def q_von_neumann(alpha, h, rho, divV): # Pressure due to the Von Neumann-Richtmyer viscosity (Monaghan 1983).

	N = len(rho)
	qq = np.zeros(N)
	
	for i in range(N):
	
		if divV[i] < 0.:
			qq[i] = alpha * rho[i] * h[i] * h[i] * divV[i] * divV[i]

	return qq


def q_bulk(alpha, h, c, rho, divV):

	N = len(rho)
	qq = np.zeros(N)
	
	for i in range(N):
		
		if divV[i] < 0.:
			qq[i] = -1. * alpha * rho[i] * h[i] * c[i] * divV[i]
	
	return qq


def q_von_bulk(alpha, h, c, rho, divV):

	q1 = q_von_neumann(alpha, h, rho, divV)
	q2 = q_bulk(alpha, h, c, rho, divV)
	
	return q1 + q2



def check_u(u, u_prevous):

	N = len(u)

	for i in range(N):
	
		if u[i] < 0.5 * u_previous[i]:
			u[i] = u_previous[i]
	
	return u




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

m = np.zeros_like(x) + 0.0002
rho = density_h(x, m, h)
rho[:50] = 1.0
rho[-20:] = 0.125

P = np.append(np.ones_like(xLeft), np.ones_like(xRight) * 0.1)

gamma = 5./3.
u = P / ((gamma - 1.) * rho)
#***********************************************

alpha = 1.0
beta = 2.0


v = np.zeros_like(x)

c = (gamma * P / rho) ** 0.5
#PI = Art_visc(x, v, rho, c, h)
#acc = Acc(x, rho, P, PI, h)
divV = divVel(x, v, h, rho, m)
qq = q_von_bulk(alpha, h, c, rho, divV)
acc = Acc_Monaghan_1st(x, rho, P, h, m, qq)

dt = 1e-4
t = 0.0

plt.ion()

#plt.figure(figsize = (10, 6))
fig, ax = plt.subplots()

for i in range(50000):

	v += acc * dt/2.0
	
	v[:50] = 0.0 # setting the velocity of the boundaries back to zero.
	v[-20:] = 0.0 # setting the velocity of the boundaries back to zero.
	
	x += v * dt
	
	rho = density_h(x, m, h)
	
	rho[:50] = 1.0 # returning the density of the boundary particles back to their initial values.
	rho[-20:] = 0.125
	
	P = (gamma - 1.0) * rho * u
	P[:50] = 1.0
	P[-20:] = 0.1
	
	c = (gamma * P / rho) ** 0.5

	divV = divVel(x, v, h, rho, m)
		
	#qq = q_von_bulk(alpha, h, c, rho, divV)
	qq1 = q_von_neumann(alpha, h, rho, divV)
	qq2 = q_bulk(alpha, h, c, rho, divV)
		
	qq = qq1
	
	ut = U_h_Monaghan_1st(x, v, rho, P, h, qq)
	
	u += ut * dt
	
	if i > 1: u = check_u(u, u_previous)
		
	
	u_previous = u.copy()

	acc = Acc_Monaghan_1st(x, rho, P, h, m, qq)
	
	v += acc * dt/2.0


	ax.cla()	
	
	ax.scatter(x, rho, s = 1)
	ax.set_title('t = ' + str(round(t, 3)))
	fig.canvas.flush_events()

	time.sleep(0.01)
	
	t += dt



