#!/usr/bin/env python3
"""
Code that creates intial conditions for the Evrard collapse

"""
# load libraries
import numpy as np  # load numpy
import matplotlib.pyplot as plt

""" initial condition parameters """

FloatType = np.float32  # double precision: np.float64, for single use np.float32
IntType = np.int32

Grid= 14 #size of the grid used to generate particle distribution
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
Vel = np.zeros((number_particles,3), dtype=FloatType)
Uthermal = np.zeros((number_particles,3), dtype=FloatType)
Mass = np.zeros((number_particles,3), dtype=FloatType)
ids = np.arange(number_particles)

Pos[:,0] = x[:,0]
Pos[:,1] = y[:,0]
Pos[:,2] = z[:,0]




Mass[:] = particle_mass

Uthermal[:] = 0.05

rad = np.sqrt(x[:]**2+y[:]**2+z[:]**2)

rho = 1.0/(2.* np.pi * rad[:])



plt.scatter(x, y, s = 0.1, color = 'k')
plt.show()


