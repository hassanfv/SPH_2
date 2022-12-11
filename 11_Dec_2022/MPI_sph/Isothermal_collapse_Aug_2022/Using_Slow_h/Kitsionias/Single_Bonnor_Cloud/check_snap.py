
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob

M_sun = 1.98892e33
UnitMass_in_g = 75. * M_sun

UnitRadius_in_cm = 1.7877e18
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3

filz = np.sort(glob.glob('./output_Gad_BEX/*.hdf5'))


j = 6

file = h5py.File(filz[j], 'r')
# ['Coordinates', 'Density', 'InternalEnergy', 'Masses', 'ParticleIDs', 'SmoothingLength', 'Velocities']

coord = file['PartType0']['Coordinates']

rho = np.array(file['PartType0']['Density']) * UnitDensity_in_cgs
print('rho = ', np.sort(rho))
print()

#Masses = file['PartType0']['Masses']
#Masses = np.array(Masses)

#Velocities = file['PartType0']['Velocities']
#np.array(Velocities)

h = file['PartType0']['SmoothingLength']
print('h = ', np.sort(h))


plt.figure(figsize = (7, 6))
plt.scatter(coord[:, 0], coord[:, 1], s = 0.1, color = 'black')

xyrange = 1.5
plt.xlim(-xyrange, xyrange)
plt.ylim(-xyrange, xyrange)

plt.savefig('1111.png')

plt.show()


