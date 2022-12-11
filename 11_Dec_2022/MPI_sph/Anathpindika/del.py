
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

res = []
for i in range(20000):

	res.append(np.random.random())


res = np.array(res)

plt.hist(res)
plt.show()
