import numpy as np
import pickle
import readchar
import matplotlib.pyplot as plt

@np.vectorize 
def cubic_spline(r, h, order):
    
    q = float(abs(r))/h
    sigma = float(2.0 / (3.0 * h))
    if order == 0:
        if q <=1.0 and q >=0.0: 
            return sigma * (1.0 - (1.5 * q * q) + (0.75 * q * q * q))
        elif q > 1.0 and q <= 2.0:
            return sigma * 0.25 * ((2.0 - q) ** 3.0)
        else:
            return 0.0
    else:
        diff_multiplier = float(np.sign(r) / h)
        if q <=1.0 and q >=0.0: 
            return float(sigma * ((-3.0 * q) + (2.25 * q * q)) * diff_multiplier)
        elif q > 1.0 and q <= 2.0:
            return float(sigma * -0.75 * ((2 - q) ** 2) * diff_multiplier)
        else:
            return 0.0



left = np.linspace(-0.5,0,320)
dxl = left[1] - left[0]
right = np.linspace(0,0.5,40)[1:]
dxr = right[1] - right[0]

n_edge = 35

left_boundary = np.linspace(-0.5 - (n_edge * dxl), -0.5 - dxl, n_edge)
right_boundary = np.linspace(0.5 + dxr, 0.5 + (n_edge * dxr), n_edge)

h = 1*(right[1] - right[0])

left = np.append(left_boundary, left)
right = np.append(right, right_boundary)

x = np.append(left, right)

rho = np.append(np.ones_like(left), np.ones_like(right)*0.125)
p = np.append(np.ones_like(left), np.ones_like(right)*0.1)
v = np.zeros_like(x)
gamma = 1.4
epsilon = 0.5
eta = 1e-04
m = 0.0015625000000000

e = p / ((gamma - 1) * rho)
num = len(x)


fig, ax = plt.subplots()

dt = 1e-03
t = 0

for j in range(300):

	print('j = ', j)

	

	x_inc = np.zeros_like(x, dtype='float32')
	rho_inc = np.copy(rho)                  #     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	v_inc = np.zeros_like(x, dtype='float32')
	e_inc = np.zeros_like(x, dtype='float32')

	## Iterate over all particles and store the accelerations in temp arrays ##
	## 10 particles to the left and 10 particles to the right as boundary ###
	for i in range(n_edge, num - n_edge):

		### Evaluating variables ###

		vij = v[i] - v
		dwij = cubic_spline(x[i] - x, h, order = 1)
		wij = cubic_spline(x[i] - x, h, order = 0)
		xij = x[i] - x

		p_rho_i = p[i] / (rho[i] * rho[i])
		p_rho_j = p / (rho * rho)

		p_rho_ij =  p_rho_i + p_rho_j

		### Artificial viscosity ###

		numerator = h * vij * xij
		denominator = (xij ** 2.0) + (eta ** 2.0)
		mu_ij = numerator / denominator
		mu_ij[mu_ij > 0] = 0.0 #only activated for approaching particles

		ci = (gamma * p[i] / rho[i]) ** 0.5
		cj = (gamma * p / rho) ** 0.5
		cij = 0.5 * (ci + cj)
		rhoij = 0.5 * (rho[i] + rho)
		numerator = (-1 * cij * mu_ij) + (mu_ij ** 2)
		denominator = rhoij
		pi_ij = numerator / denominator

		### Evaluating gradients ###

		grad_v = -1 * np.sum(m * (p_rho_ij + pi_ij) * dwij)
		grad_e = 0.5 * np.sum(m * (p_rho_ij + pi_ij) * vij * dwij)

		rho_inc[i] = np.sum(m * wij) #dt * grad_rho
		v_inc[i] = dt * grad_v
		e_inc[i] = dt * grad_e

		x_inc[i] = dt * v[i]
		
		#vij_res.append(vij)

	    
	## Update the original arrays using the increment arrays ##
	
	#plt.scatter(np.arange(len(rho_inc)), rho_inc, s = 1)
	#plt.show()
	#s()
	    
	rho = rho_inc
	v += v_inc
	e += e_inc

	### Update pressure ###

	p = (gamma - 1) * e * rho

	### Update positions ###

	x += x_inc


	ax.cla()

	ax.scatter(x, rho, s = 3, color = 'k')
	#ax.axis(ymin = -0.1, ymax = 1.5)
	plt.title('t = ' + str(t))
	plt.draw()
	plt.pause(0.001)

	#kb = readchar.readkey()
	
	t += dt
	print('t = ', t)
	print()
	
	


plt.savefig('Fig.png')

plt.show()

data = dict()
data['rho'] = rho
data['p'] = p
data['v'] = v
data['x'] = x
data['e'] = e

with open('shocktube_ce_outputX.pkl', 'wb') as f:
	pickle.dump(data, f)
	

