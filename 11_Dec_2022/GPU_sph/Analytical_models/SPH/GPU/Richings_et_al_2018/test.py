
# Here the Gas and DM have similar number of particles!

import numpy as np
import matplotlib.pyplot as plt
import time
from libsx import *
from shear import *
import pandas as pd
import pickle


# Constants
Msun = 1.989e33  # Solar mass in grams
G = 6.67430e-8  # Gravitational constant in cm^3 g^-1 s^-2
kB = 1.380649e-16  # Boltzmann constant in erg K^-1
mH = 1.6726219e-24  # Proton mass in grams
clight = 29979245800. # cm/s


nH = 10.0
mu = 0.61 # fully ionized gas with solar metallicity!
rho = mu * mH * nH

x = 0.6 # kpc
x = x * 3.086e21  # kpc to cm

M = x*x*x * rho

print(f'Mass = {M/Msun:.3E} M_sun')

mSPH = 80.0 # M_sun

print(f'For mSPH = {mSPH} M_sun, {M/Msun:.2E} corresponds to {M/Msun/mSPH:.2E} gas particles!')



