
import numpy as np
import time
from libsx import *
import pickle
import struct

np.random.seed(42)

# Constants
Msun = 1.989e33  # Solar mass in grams
G = 6.67430e-8  # Gravitational constant in cm^3 g^-1 s^-2
kB = 1.380649e-16  # Boltzmann constant in erg K^-1
mH = 1.6726219e-24  # Proton mass in grams
clight = 29979245800.  # cm/s
cm_to_kpc = 3.086e21

sigma = 200. * 1000. * 100. # cm/s =====> 200 km/s - See eq.1 in Richings et al - 2018

# In the current simulation all these particles are gas particle. We later add a collisionless BH.
N_particles = 300000  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

nH = 0.1  # cm^-3  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
T_gas = 1e4  # K

mu = 0.61  # fully ionized gas with solar metallicity!
#rho = mu * mH * nH
XH = 0.7
rho = mH * nH / XH

mSPH = 20.0  # M_sun # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
mSPH_in_g = mSPH * Msun  # grams
M_tot = N_particles * mSPH
M_tot_in_g = M_tot * Msun  # Total mass in grams.

L_box_in_cm = (M_tot_in_g / rho)**(1./3.)  # box size in cm.

L_box_in_kpc = L_box_in_cm / cm_to_kpc

print(f'With nH = {nH}, and N_particles = {N_particles}, we get L_box = {round(L_box_in_kpc, 2)} kpc.')

# ------ Setting the units of the simulation ---------
grav_const_in_cgs = 6.67430e-8
unitMass_in_g = M_tot_in_g
unitLength_in_cm = cm_to_kpc  # This is 1 kpc in cm so our unit length is 1 kpc!
unitTime_in_s = (unitLength_in_cm**3/grav_const_in_cgs/unitMass_in_g)**0.5
unitVelocity_in_cm_per_s = unitLength_in_cm / unitTime_in_s
Unit_u_in_cgs = grav_const_in_cgs * unitMass_in_g / unitLength_in_cm
UnitDensity_in_cgs = unitMass_in_g / unitLength_in_cm**3

print()
print(f'Unit Length = {unitLength_in_cm} cm')
print(f'Unit mass = {unitMass_in_g} grams')
print(f'Unit_time_in_s = {unitTime_in_s:.3E} seconds')
print(f'Unit_time in kyrs = {round(unitTime_in_s/3600./24./365.25/1000., 2)} kyrs')
print(f'Unit_time in Myrs = {round(unitTime_in_s/3600./24./365.25/1e6, 4)} Myrs')
print(f'unitVelocity_in_cm_per_s = {round(unitVelocity_in_cm_per_s, 2)} cm/s')
print(f'unitVelocity_in_km/s = {round(unitVelocity_in_cm_per_s/100000., 2)} km/s')
print(f'Unit_u_in_cgs = {Unit_u_in_cgs:.4E}')
print(f'UnitDensity_in_cgs = {UnitDensity_in_cgs:.3E}')
print()
# ------------------------------------------------------


# Saving the parameters and constants!
G = 1.0

Tou_in = 1.0
L_AGN = 1e46  # erg/s !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Note that we multiply u by mass because u is in per unit mass!!
L_AGN_code_unit = L_AGN / (Unit_u_in_cgs * unitMass_in_g / unitTime_in_s)
# So L_AGN is now in energy per unit mass per unit time.

clight_code_unit = clight / unitVelocity_in_cm_per_s

vin = 30000. * 1000. * 100.
vin_in_code_unit = vin / unitVelocity_in_cm_per_s

M_dot_in_code_unit = Tou_in * L_AGN_code_unit / clight_code_unit / vin_in_code_unit

u_for_10K_Temp = (3/2) * kB * T_gas / mu / mH / Unit_u_in_cgs

m_sph_high_res = mSPH_in_g / unitMass_in_g
m_sph_high_res /= 1.0 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

sigma_in_code_unit = sigma / unitVelocity_in_cm_per_s

print()
print('M_dot_in_code_unit = ', M_dot_in_code_unit)
print()

mSPH = mSPH_in_g / M_tot_in_g

m_outflow_particles = 10 * mSPH ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

dt_code = 2e-7

mass_to_inject_code = M_dot_in_code_unit * dt_code

num_particles_ejected_per_timeStep = mass_to_inject_code / m_outflow_particles

print()
print(f'num_particles_ejected_per_timeStep = {num_particles_ejected_per_timeStep:.2f}')





