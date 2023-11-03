import struct
import matplotlib.pyplot as plt
import numpy as np


filename = "data.bin"

with open(filename, "rb") as f:
    # Read N
    N = struct.unpack("i", f.read(4))[0]
    
    # Read x, y, z arrays
    x = np.array(struct.unpack(f"{N}f", f.read(N*4)))
    y = np.array(struct.unpack(f"{N}f", f.read(N*4)))
    z = np.array(struct.unpack(f"{N}f", f.read(N*4)))
    
    # Read nCell
    nCellLength = struct.unpack("i", f.read(4))[0]
    nCell = list(struct.unpack(f"{nCellLength}i", f.read(nCellLength*4)))


# Open the file in read mode
with open('n100.txt', 'r') as file:
    # Read lines and convert them to integers
    n100 = [int(line.strip()) for line in file]


xx = [0.388759]
yy = [0.822705]
zz =  [1.399024]

plt.figure(figsize = (7, 7))
plt.scatter(x, y, s = 2, color = 'k')
plt.scatter(x[nCell], y[nCell], s = 30, color = 'lime')

plt.scatter(x[n100], y[n100], s = 30, color = 'yellow')

plt.scatter(xx, yy, s = 60, color = 'red')



plt.show()



