
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time
from numba import jit, njit


def get_coeff(t):

	x1 = 1.2165
	y1 = 0.5
	x2 = 1.2701
	y2 = 0.1
	slope = (y1-y2)/(x1-x2)
	coeff = y1 + slope * (t - x1)
	
	return coeff



#===== getPE
@njit
def getPE(pos, m, G, epsilon):

	N = pos.shape[0]

	dx = np.empty(3)
	PE = 0.0

	for i in range(N):
		for j in range(i+1, N):
			
			dx = pos[i, 0] - pos[j, 0]
			dy = pos[i, 1] - pos[j, 1]
			dz = pos[i, 2] - pos[j, 2]
			
			rr = (dx**2 + dy**2 + dz**2)**0.5				
			
			fk = 0.0
			
			if rr != 0.0 :
				inv_r = 1.0 / rr

			epsilonij = 0.5 * (epsilon[i] + epsilon[j])
			q = rr / epsilonij
			
			if (q <= 1.0) & (q != 0.0):
				fk = m[j] * ((-2.0/epsilonij) * ( (1.0/3.0)*q**2 - (3.0/20.0)*q**4 + (1.0/20.0)*q**5 ) + 7.0/5.0/epsilonij)

			if (q > 1.) and (q <= 2.):
				fk = m[j]*((-1.0/15.0)*inv_r - (1.0/epsilonij) * ((4.0/3.0)*q**2 - q**3 + (3.0/10.0)*q**4 - (1.0/30.0)*q**5) + 8.0/5.0/epsilonij)

			if q > 2.:
				fk = m[j] * inv_r

			PE -= G * m[i] * fk

	return PE




#unitTime_in_Myr =  0.07673840095663824 # Myr

M_sun = 1.98992e+33 # gram
UnitMass_in_g = 1.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
UnitRadius_in_cm = 4.95e+16  #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3

G = 6.67259e-8 #  cm3 g-1 s-2

print(f'UnitDensity_in_cgs = {UnitDensity_in_cgs} g/cm^3')

rho0 = 3.82e-18 # g/cm^3


#filz = np.sort(glob.glob('../UB/Outputs_20k/*.pkl'))

#filz = np.sort(glob.glob('/mnt/Linux_Shared_Folder_2022/Outputs_60k_UB/*.pkl'))
filz = np.sort(glob.glob('./Outputs_60k_UB_h/Outputs/*.pkl'))

plt.ion()
fig, ax = plt.subplots(figsize = (6, 5))

#print(filz)

kb = ''

res = []

for j in range(2000, len(filz), 20):  # 35.54 + 1 = 36.54

	print('j = ', j)

	with open(filz[j], 'rb') as f:
		data = pickle.load(f)

	r = data['pos']
	h = data['h']
	m = data['m']
	v = data['v']
	print(r.shape)
	
	print('h = ', np.sort(h))

	x = r[:, 0]
	y = r[:, 1]
	z = r[:, 2]
	t = data['current_t']
	rho = data['rho']
	unitTime_in_kyr = data['unitTime'] /3600./24./365.25/1000.
	t_ff = data['t_ff']
	unitVelocity = data['unitVelocity']
	
	t_t_ff = round(t*data['unitTime']/t_ff, 4) # t/t_ff
	
	rho = rho*UnitDensity_in_cgs
	
	nyneg = np.where(y < 10000.0)[0]
	
	xneg = x[nyneg]
	yneg = y[nyneg]
	zneg = z[nyneg]
	
	rhoneg = rho[nyneg]
	mneg = m[nyneg]
	hneg = h[nyneg]
	vneg = v[nyneg]
	
	n_rho_max_1 = np.where(rhoneg == max(rhoneg))[0]
	xmax1 = xneg[n_rho_max_1]
	ymax1 = yneg[n_rho_max_1]
	zmax1 = zneg[n_rho_max_1]
	
	#--- coord. of the fragment particles
	#nfrag1 = np.where(rhoneg >= get_coeff(t_t_ff)*max(rhoneg))[0]
	
	#nfrag1 = np.where(rhoneg >= 100*rho0)[0]
	
	nfrag1 = np.where(rhoneg >= 0.10*max(rhoneg))[0]
	
	xfrag1 = xneg[nfrag1].reshape((-1, 1))
	yfrag1 = yneg[nfrag1].reshape((-1, 1))
	zfrag1 = zneg[nfrag1].reshape((-1, 1))
	
	mfrag1 = mneg[nfrag1]
	rhofrag1 = rhoneg[nfrag1]
	hfrag1 = hneg[nfrag1]
	
	vfrag1 = vneg[nfrag1]
	
	vxfrag1 = vfrag1[:, 0]
	vyfrag1 = vfrag1[:, 1]
	vzfrag1 = vfrag1[:, 2]

	epsilonfrag1 = hfrag1.copy()
	
	print(nfrag1.shape)
	print('coeff = ', get_coeff(t_t_ff))
	
	
	#--- Calculation the grav. PE -----
	rfrag1 = np.hstack((xfrag1, yfrag1, zfrag1))
	
	PE = getPE(rfrag1*UnitRadius_in_cm, mfrag1*UnitMass_in_g, G, epsilonfrag1*UnitRadius_in_cm)
	
	print('PE = ', PE)
	
	#--- Calculating rotational Energy -----
	
	Erot = 0.0
	for i in range(len(xfrag1)):
	
		rtmp = (xfrag1[i]**2 + yfrag1[i]**2)**0.5
		
		vphi = (xfrag1[i]*vyfrag1[i] - yfrag1[i]*vxfrag1[i]) / rtmp
		
		Erot += 0.5 * mfrag1[i]*UnitMass_in_g * (vphi*unitVelocity)**2

	print('Erot = ', Erot)
	print('Erot/PE = ', Erot/PE)
	print('M_frag = ', np.sum(mfrag1))
	print('t/t_ff = ', t_t_ff)
	print()

	
	res.append([t_t_ff, np.abs(Erot/PE)])
	
	
	if True:
		ax.cla()

		ax.scatter(x, y, s = 0.01, color = 'black')
		ax.scatter([xmax1, xmax1], [ymax1, ymax1], color = 'lime', s = 20) # Max rho marker.
		ax.scatter(xfrag1, yfrag1, s = 0.1, color = 'red')
		
		xyrange = 0.16
		
		ax.axis(xmin = -xyrange, xmax = xyrange)
		ax.axis(ymin = -xyrange, ymax = xyrange)

		
		ax.set_title('t = ' + str(np.round(t*unitTime_in_kyr,2)) + '       t/t_ff = ' + str(t_t_ff))
		fig.canvas.flush_events()
		time.sleep(0.01)
		
		kb =readchar.readkey()
		
		if kb == 'q':
			break

plt.savefig('1111.png')







