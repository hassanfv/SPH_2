
import numpy as np
import pickle
import glob
import pandas as pd


with open('00027.pkl', 'rb') as f:
	data = pickle.load(f)


r = data['pos']
h = data['h']

x = r[:, 0]
y = r[:, 1]
z = r[:, 2]

dictx = {'x': x, 'y': y, 'z': z, 'h': h}

df = pd.DataFrame(dictx)

df.to_csv('data.csv', index = False)







