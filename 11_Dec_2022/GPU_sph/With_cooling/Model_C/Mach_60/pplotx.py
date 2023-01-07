
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./Outputs/G-0.000067.csv')

rho = df['rho'].values
h = df['h'].values


x = [1, 0] # x-y plane

#x = [0, 1] # y-z plane

jj = 153450

if x[0]:
	plt.figure(figsize = (11, 5))
	plt.scatter(df['x'].values, df['y'].values, s = 0.001, color = 'k')
	plt.scatter(df['x'].values[jj], df['y'].values[jj], s = 20, color = 'red')
	plt.xlim(-2.2, 2.2)
	plt.ylim(-1.2, 1.2)
	#plt.axvline(x = -0.0202, linestyle = '--')


if x[1]:
	plt.figure(figsize = (6, 5))
	plt.scatter(df['y'].values, df['z'].values, s = 0.001, color = 'k')
	plt.scatter(df['y'].values[jj], df['z'].values[jj], s = 20, color = 'red')
	xyrange = 1.0
	plt.xlim(-xyrange, xyrange)
	plt.ylim(-xyrange, xyrange)



# Find intresting particles !!!

x = df['x'].values
y = df['y'].values
z = df['z'].values


for i in range(len(x)):

	if (np.abs(x[i]) < 0.04) & (np.abs(x[i]) > 0.02):
	
		if np.abs(y[i]) < 0.15:
		
			if np.abs(z[i]) < 0.05:
			
				print(i)


plt.show()




