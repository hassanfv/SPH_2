
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import time
import pandas as pd



with open('./Outputs/00001.pkl', 'rb') as f:
	data = pickle.load(f)

hprevious = data['h']


with open('./Outputs/00002.pkl', 'rb') as f:
	data = pickle.load(f)


r = data['pos']
h = data['h']
v = data['v']

print(r.shape)

print('hprevious = ', np.sort(hprevious))
print()
print('h (expected) = ', np.sort(h))
print()

x = r[:, 0]
y = r[:, 1]
z = r[:, 2]

dictx = {'x': x, 'y': y, 'z': z, 'h': hprevious}

df = pd.DataFrame(dictx)

df.to_csv('test_data.csv', index = False, header = False)


