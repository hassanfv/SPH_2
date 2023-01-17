import numpy as np


with open('test3.out' , 'r') as f:

	k = 0
	for x in f:
			
		if 'MolWgt' in x:
		
			if not (k%2):
				print(x[105:113])
			k += 1











