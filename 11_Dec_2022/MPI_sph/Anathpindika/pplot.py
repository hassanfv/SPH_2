
import pickle
import matplotlib.pyplot as plt


with open('Data.pkl', 'rb') as f:
	data = pickle.load(f)

r = data['r']

x = r[:, 0]
y = r[:, 1]
z = r[:, 2]


plt.figure(figsize = (14, 6))
plt.scatter(x, y, s = 0.1, color = 'k')
plt.show()





