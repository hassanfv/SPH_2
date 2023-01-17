import numpy as np
import matplotlib.pyplot as plt


checker = 0

res = []

with open('test2.het', 'r') as f:

	for i in range(5000):
		x = f.readline()
		
		if x != '':
		
			if checker == 1:
				#print(x[34:44]) # cooling rate
				#print(x[23:33]) # heating rate
				#print(x[12:22]) # temperature
				res.append([float(x[12:22]), float(x[23:33]), float(x[34:44])])
				checker = 0
			
			if 'GR' in x:
				checker = 1

res = np.array(res)

T = res[:, 0]
heat = res[:, 1]
cool = res[:, 2]
		

plt.plot(np.log10(T), np.log10(cool), color = 'blue')
plt.plot(np.log10(T), np.log10(heat), color = 'red')

plt.xlim(2, 8)
plt.ylim(-25, -18.0)

plt.show()





