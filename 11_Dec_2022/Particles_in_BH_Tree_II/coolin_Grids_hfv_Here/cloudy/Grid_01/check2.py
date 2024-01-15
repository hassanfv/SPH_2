import numpy as np
import matplotlib.pyplot as plt

hden_beg = -4.0
hden_end = 3.0 + 1e-6 # 1e-6 is added so that 3.0 itself is counted
hden_stp = 0.2

hdenArr = np.arange(hden_beg, hden_end, hden_stp)
N_hden = len(hdenArr)

print(f'N_hden = {N_hden}')

with open('test3.het', 'r') as f:

	for j in range(N_hden):

		checker = 0
		res = []

		for i in range(5000): # 5000 is arbitrary.. it just needs to be large enough. 5000 is more than enough!
			x = f.readline()
			
			if x != '':
			
				if checker == 1:
					res.append([float(x[12:22]), float(x[23:33]), float(x[34:44])])
					checker = 0
					if float(x[12:22]) == 1.0000e+08:
						break
				
				if ('GR' in x) | ('erg' in x): # this is used only for the first temperature in the beginning of the grid. 
					checker = 1


		resx = np.array(res)

		T = resx[:, 0]
		heat = resx[:, 1]
		cool = resx[:, 2]
				

		plt.plot(np.log10(T), np.log10(cool), color = 'blue')
		plt.plot(np.log10(T), np.log10(heat), color = 'red')

		plt.xlim(2, 8)
		#plt.ylim(-25, -18.0)

	plt.show()





