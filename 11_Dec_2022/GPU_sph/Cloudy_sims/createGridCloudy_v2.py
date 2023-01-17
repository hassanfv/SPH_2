import numpy as np
import pandas as pd

# The difference with previous version is that here we include mu (i.e. mean molecular weight).


#============ Gathering the mu (i.e. Mean Molecular Weight) values from *.out file:
with open('test.out' , 'r') as f:

	k = 0
	muRes = []
	
	for x in f:
			
		if 'MolWgt' in x:
		
			if not (k%2): # we take the first value (as we have two iterations !)
			
				muRes.append(x[105:113])
			k += 1
#====== End of mu collection ! ==============================


hden_beg = -4.0 # in log
hden_end = 3.0 + 1e-6 # 1e-6 is added so that 3.0 itself is counted
hden_stp = 0.02

hdenArr = np.arange(hden_beg, hden_end, hden_stp)
N_hden = len(hdenArr)

T_beg = 2.0 # in log
T_end = 8.0 + 1e-6 # in log
T_stp = 0.02 # in log
TempArr = np.arange(T_beg, T_end, T_stp)
N_Temp = len(TempArr)

print(f'N_hden = {N_hden}')
print(f'N_TempArr = {N_Temp}')

s()

res = []
k = 0

with open('test3.het', 'r') as f:

	for j in range(N_hden):

		checker = 0

		for i in range(5000): # 5000 is arbitrary.. it just needs to be large enough. 5000 is more than enough!
			x = f.readline()
			
			if x != '':
			
				if checker == 1:

					res.append([float(x[12:22]), 10**hdenArr[j], muRes[k], float(x[23:33]), float(x[34:44])])
					checker = 0
					k += 1

					if float(x[12:22]) == 1.0000e+08:
						break
				
				if ('GR' in x) | ('erg' in x): # ('erg' in x) is used only for the first temperature in the beginning of the grid. 
					checker = 1


		resx = np.array(res)
		
		df = pd.DataFrame(resx, columns = ['T', 'nH', 'mu', 'heating', 'cooling'])
		
		df.to_csv('CloudyCoolingGridX.csv', index = False)





