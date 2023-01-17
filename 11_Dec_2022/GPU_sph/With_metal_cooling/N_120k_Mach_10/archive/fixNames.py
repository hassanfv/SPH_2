
import os
import glob
import numpy as np


PATH = './Outputs/'

filz = np.sort(glob.glob(PATH + '*.csv'))

for j in range(len(filz)):

	#t = float(filz[j].split('/')[-1][8:-4])
	#t = float(filz[j].split('/')[-1][8:-4])

	if len(filz[j]) == 32:
		nam = filz[j] # './Outputs/G-00000000.000000.csv'

		a = nam.split('/')[-1][9:]

		a = 'G-' + a

		os.rename(nam, PATH + a)





