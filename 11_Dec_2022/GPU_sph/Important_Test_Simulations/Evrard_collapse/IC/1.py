#!/usr/bin/env python3
"""
Code that creates intial conditions for the Evrard collapse

"""
# load libraries
import numpy as np  # load numpy
import h5py    # hdf5 format 
import matplotlib.pyplot as plt
import pickle

""" initial condition parameters """
FilePath = 'IC.hdf5'

FloatType = np.float32  # double precision: np.float64, for single use np.float32
IntType = np.int32

Grid= 74 #size of the grid used to generate particle distribution   # 14 corresponds to 1472 particles.
Mtot= 1.0 #total mass of the sphere
gamma = 5./3.

x = np.zeros((Grid,Grid,Grid))
y = np.zeros((Grid,Grid,Grid))
z = np.zeros((Grid,Grid,Grid))
vx = np.zeros((Grid,Grid,Grid))
vy = np.zeros((Grid,Grid,Grid))
vz = np.zeros((Grid,Grid,Grid))
m = np.zeros((Grid,Grid,Grid))
u = np.zeros((Grid,Grid,Grid))

xx = ((np.arange(Grid)-(Grid/2-0.5))/(Grid/2.0))

for i in range(Grid):
	for j in range(Grid):
		x[:,i,j] = xx[:]
		y[i,:,j] = xx[:]
		z[i,j,:] = xx[:]

x = x.flatten()
y = y.flatten()
z = z.flatten()


#stretching initial conditons to get 1/r density distribution
r = np.sqrt(x**2+y**2+z**2)**0.5

x=x*r
y=y*r
z=z*r

rad = np.sqrt(x**2+y**2+z**2)

j = np.argwhere(rad < 1.0)



number_particles = j.size

particle_mass = Mtot / number_particles

print("We use "+str(number_particles)+ " particles")


x = x[j]
y = y[j]
z = z[j]


Pos = np.zeros((number_particles,3), dtype=FloatType)
#Vel = np.zeros((number_particles,3), dtype=FloatType)
#Uthermal = np.zeros((number_particles,3), dtype=FloatType)
#Mass = np.zeros((number_particles,3), dtype=FloatType)
#ids = np.arange(number_particles)

Pos[:,0] = x[:,0]
Pos[:,1] = y[:,0]
Pos[:,2] = z[:,0]

plt.scatter(Pos[:, 0], Pos[:, 1], s = 0.5, color = 'k')
plt.show()


###################
G = 1.0
NSample = Pos.shape[0]
Mcld_in_g = 0.0
Rcld_in_cm = 0.0
Rcld_in_pc = 0.0 #Rcld_in_cm/3.086e18
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2
muu = 3.0
gamma = 5./3.
c_0 = 0.0 # cm/s # this corresponds to c_iso in Arrega et al - 2007
Mach = 0.0 # not applicable in this test.

masses = np.zeros(NSample) + 1.0 / NSample

#---- physical parameters used for GPU ----
#            0      1     2         3          4           5       6    7            8          9
paramz = [NSample, c_0, gamma, Rcld_in_pc, Rcld_in_cm, Mcld_in_g, muu, Mach, grav_const_in_cgs, G]

u = 0.05 + np.zeros(len(masses))  #np.zeros(len(masses)) + c_0**2 / unitVelocity_in_cm_per_s**2

vel = np.zeros_like(Pos)

dictx = {'r': Pos, 'v': vel, 'm': masses, 'u': u, 'paramz': paramz}

with open('tmp1.pkl', 'wb') as f:
	pickle.dump(dictx, f)

print()
print(f'Total number of particles = {Pos.shape[0]}')
print()
print('********************************')
print('Step_1 Successfully Finished !!!')
print('********************************')


