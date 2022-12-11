
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob


filz = np.sort(glob.glob('./Outputs/*.pkl'))
filz = np.sort(glob.glob('/mnt/Linux_Shared_Folder_2022/Outputs_Model_3_Anathpindika_2009/*.pkl'))

j = 670

with open(filz[j], 'rb') as f:
	data = pickle.load(f)


r = data['pos']

print(r.shape)

x = r[:, 0]
y = r[:, 1]
z = r[:, 2]

plt.figure(figsize = (9, 8))

plt.scatter(x, y, s = 0.1, color = 'black')
xyrange = 2.
plt.xlim(-xyrange, xyrange)
plt.ylim(-xyrange, xyrange)
plt.show()








