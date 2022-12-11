
import pickle
import numpy as np

with open('u_after_cool_MPI.pkl', 'rb') as f:
	res1 = pickle.load(f)

u1 = res1[:, 2]

with open('u_after_cool.pkl', 'rb') as f:
	res2 = pickle.load(f)

u2 = res2[:, 2]

print(u1.shape, u1.shape)
print(np.sum(u1-u2))










