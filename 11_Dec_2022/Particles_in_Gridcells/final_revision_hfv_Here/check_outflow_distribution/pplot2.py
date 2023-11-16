import numpy as np
import matplotlib.pyplot as plt



# Read the binary file
with open("data.bin", "rb") as f:
    # Read N
    N = np.fromfile(f, dtype=np.int32, count=1)[0]
    
    # Read arrays
    x = np.fromfile(f, dtype=np.float32, count=N)
    y = np.fromfile(f, dtype=np.float32, count=N)
    z = np.fromfile(f, dtype=np.float32, count=N)
    vx = np.fromfile(f, dtype=np.float32, count=N)
    vy = np.fromfile(f, dtype=np.float32, count=N)
    vz = np.fromfile(f, dtype=np.float32, count=N)


#plt.figure(figsize = (8, 8))

vv = np.sqrt(vx*vx + vy*vy + vz*vz)

print(np.sort(vv))

plt.hist(vv, bins = 20)

plt.show()


