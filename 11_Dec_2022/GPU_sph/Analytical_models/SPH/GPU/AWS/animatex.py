
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import readchar


def read_arrays_from_binary(filename):
    # Read the binary file
    with open(filename, 'rb') as file:
        # Read N and NG from the file
        N = np.frombuffer(file.read(4), dtype=np.int32)[0]
        NG = np.frombuffer(file.read(4), dtype=np.int32)[0]

        # Read the arrays from the file
        x = np.frombuffer(file.read(N * 4), dtype=np.float32)
        y = np.frombuffer(file.read(N * 4), dtype=np.float32)
        z = np.frombuffer(file.read(N * 4), dtype=np.float32)
        vx = np.frombuffer(file.read(N * 4), dtype=np.float32)
        vy = np.frombuffer(file.read(N * 4), dtype=np.float32)
        vz = np.frombuffer(file.read(N * 4), dtype=np.float32)
        rho = np.frombuffer(file.read(NG * 4), dtype=np.float32)
        h = np.frombuffer(file.read(NG * 4), dtype=np.float32)
        u = np.frombuffer(file.read(NG * 4), dtype=np.float32)

    return x, y, z, vx, vy, vz, rho, h, u, N, NG


#with open('PlaneXY.pkl', 'rb') as f:
#	nXY = pickle.load(f) # the indices of the particles residing in the XY plane at Z = 0.0

dirx = './Outputs/'

filez = np.sort(os.listdir(dirx))

plt.ion()

fig, ax = plt.subplots(figsize = (6, 5))



for j in range(0, len(filez), 1):

	print(j)
	
	x, y, z, vx, vy, vz, rho, h, u, N, NG = read_arrays_from_binary(dirx + filez[j])
	
	xx = x[:NG]
	yy = y[:NG]
	zz = z[:NG]

	nz = np.where(np.abs(zz) <= 0.04)[0]

	plt.scatter(xx[nz], yy[nz], s = 0.02, color = 'k')

	x = y = 1.0

	ax.axis(xmin = -x, xmax = x)
	ax.axis(ymin = -y, ymax = y)

	fig.canvas.flush_events()
	time.sleep(0.01)
	
	ax.cla()
	
	kb = readchar.readkey()
	
	if kb == 'q':
		break



