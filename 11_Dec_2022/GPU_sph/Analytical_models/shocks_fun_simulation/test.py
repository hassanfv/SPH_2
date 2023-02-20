
import numpy as np
import matplotlib.pyplot as plt
import readchar
import time


N = 100

coeff = np.zeros(N)
coeff[0] = 1

vsnd = 2.0
vsrc = 4.0

dt = 1.0

plt.ion()
fig, ax = plt.subplots(1, figsize = (8, 8))

theta = np.linspace( 0 , 2 * np.pi , 150 )

j = 0

while j < N - 3:

	nx = np.where(coeff != 0)[0]

	coeff_tmp = coeff[nx]
	
	R_t = vsnd * coeff_tmp * dt
	x_t = vsrc * coeff_tmp * dt
	
	ax.cla()
	
	for i in range(len(R_t)):
		radius = R_t[i]	
		
		a = radius * np.cos(theta) - x_t[i]
		b = radius * np.sin(theta)
		 
		ax.plot(a, b)
		
	#---- Thick circle -----
	#radius = R_t[0]
	#a = radius * np.cos(theta) - x_t[0]
	#b = radius * np.sin(theta)
		 
	#ax.plot(a, b, color = 'r', linewidth = 5)
	#-----------------------
	
	fig.canvas.flush_events()
	time.sleep(0.01)

	print('nx = ', nx)
	print('coeff Before = ', coeff)
	coeff[nx] += 1
	coeff[nx[-1]+1] = 1
	print('coeff After = ', coeff)
	
	kb = readchar.readkey()
	
	if kb == 'q':
		break

plt.savefig('img.png')
	
