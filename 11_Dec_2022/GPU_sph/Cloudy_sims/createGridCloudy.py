import numpy as np
import pandas as pd

hden_beg = -4.0 # in log
hden_end = 3.0 + 1e-6 # 1e-6 is added so that 3.0 itself is counted
hden_stp = 0.1

hdenArr = np.arange(hden_beg, hden_end, hden_stp)
N_hden = len(hdenArr)

T_beg = 2.0 # in log
T_end = 8.0 + 1e-6 # in log
T_stp = 0.1 # in log
TempArr = np.arange(T_beg, T_end, T_stp)
N_Temp = len(TempArr)

print(f'N_hden = {N_hden}')
print(f'TempArr = {N_Temp}')

res = []

with open('test3.het', 'r') as f:

	for j in range(N_hden):

		checker = 0

		for i in range(5000): # 5000 is arbitrary.. it just needs to be large enough. 5000 is more than enough!
			x = f.readline()
			
			if x != '':
			
				if checker == 1:
					res.append([float(x[12:22]), 10**hdenArr[j], float(x[23:33]), float(x[34:44])])
					checker = 0
					if float(x[12:22]) == 1.0000e+08:
						break
				
				if ('GR' in x) | ('erg' in x): # ('erg' in x) is used only for the first temperature in the beginning of the grid. 
					checker = 1


		resx = np.array(res)
		
		df = pd.DataFrame(resx, columns = ['T', 'rho', 'heating', 'cooling'])
		
		df.to_csv('CloudyCoolingGrid.csv', index = False)





