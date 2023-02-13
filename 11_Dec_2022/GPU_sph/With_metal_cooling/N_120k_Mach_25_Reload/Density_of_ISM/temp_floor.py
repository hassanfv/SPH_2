
import numpy as np


rho = 1e-22
mu = 0.6
mH = 1.6726e-24
G = 6.6738e-8
kB = 1.3807e-16


Nngb = 64
mSPH = 4e-6 * 7.315e35

T_floor = rho**(1./3.) * mu * mH * G / np.pi / kB * (Nngb * mSPH)**(2./3.)

print(T_floor)



