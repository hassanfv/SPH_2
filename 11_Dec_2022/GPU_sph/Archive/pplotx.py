
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./Outputs/G-0.001790.csv')

rho = df['rho'].values
h = df['h'].values


x = [1, 0] # x-y plane

#x = [0, 1] # y-z plane

jj = 13500

if x[0]:
	plt.figure(figsize = (11, 5))
	plt.scatter(df['x'].values, df['y'].values, s = 0.001, color = 'k')
	plt.scatter(df['x'].values[jj], df['y'].values[jj], s = 20, color = 'red')
	plt.xlim(-2.2, 2.2)
	plt.ylim(-2, 2)
	#plt.axvline(x = -0.0202, linestyle = '--')


if x[1]:
	plt.figure(figsize = (6, 5))
	plt.scatter(df['y'].values, df['z'].values, s = 0.001, color = 'k')
	xyrange = 1.0
	plt.xlim(-xyrange, xyrange)
	plt.ylim(-xyrange, xyrange)



plt.show()
