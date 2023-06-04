
import numpy as np
import matplotlib.pyplot as plt

filename = "outMultiXX.bin"
#filename = "outSingle.bin"
N = 1050000  # Replace with actual size

# Open the file in binary mode
with open(filename, "rb") as f:
    # Read the accx, accy and accz arrays
    accx = np.fromfile(f, dtype=np.float32, count=N)
    accy = np.fromfile(f, dtype=np.float32, count=N)
    accz = np.fromfile(f, dtype=np.float32, count=N)



print(np.sum(accx != 0))

plt.hist(accx, bins = np.arange(-50, 50, 1))
plt.show()
