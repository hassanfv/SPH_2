
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time
import pickle
from libsx2_2t import *
import time
import h5py
import pandas as pd

np.random.seed(42)

NSample = 100000 # The desired number of particles to be included in ONE cloud in the IC.
NSample = int(2 * NSample)

with open('clouds_106k_RAND.pkl', 'rb') as f:
	data = pickle.load(f)

# These contain data for both colliding clouds!
r = data['r']

Ntot = r.shape[0]
aa = np.arange(Ntot)
np.random.shuffle(aa)
rows = aa[:NSample]

r = data['r'][rows, :]
vel = data['v'][rows, :]
m = data['m'][rows]
# Normalizing m so that sum(m) = 2 again (note we have 2 clouds!!!)!
m = 2.0 * m / np.sum(m)
rho_cen = data['rho_cen']

rho = data['rho']
rho = np.concatenate((rho, rho))
rho = rho[rows]

Rcld_in_pc = data['Rcld_in_pc']
Rcld_in_cm = data['Rcld_in_cm']
Mcld_in_g = data['Mcld_in_g']
muu = data['mu']
gamma = data['gamma']
Mach = data['Mach']
grav_const_in_cgs = data['grav_const_in_cgs']
c_0 = data['c_0']

Rcld_in_pc = Rcld_in_cm/3.086e18

dictx = {'r': r, 'v': vel, 'm': m, 'rho_cen': rho_cen, 'c_0': c_0, 'gamma': gamma,
	 'Rcld_in_pc': Rcld_in_pc, 'Rcld_in_cm': Rcld_in_cm, 'Mcld_in_g': Mcld_in_g, 'mu': muu, 'Mach': Mach} # rho_cen = central density.

num = str(int(NSample/2/1000)) # Used for out put file name.

with open('IC_Anathpin_' + num + 'k_RAND.pkl', 'wb') as f:
	pickle.dump(dictx, f)
#----------------------------


#***********************************************
#**************** FOR GPU **********************
#***********************************************
UnitMass_in_g = Mcld_in_g
UnitRadius_in_cm = Rcld_in_cm

unitVelocity = (grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm)**0.5

x = np.round(r[:, 0], 5)

y = np.round(r[:, 1], 5)
z = np.round(r[:, 2], 5)

vx = np.round(vel[:, 0], 5) / unitVelocity
vy = np.round(vel[:, 1], 5) / unitVelocity
vz = np.round(vel[:, 2], 5) / unitVelocity
	
dictx = {'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz, 'm': m}

df = pd.DataFrame(dictx)

df.to_csv('GPU_IC_Anathpin_RAND_tmp.csv', index = False)

#M_sun = 1.98992e+33 # gram
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2
G = 1.0

#---- Writing physical parameters to a file (for GPU) ----
paramz = [NSample, c_0, gamma, Rcld_in_pc, Rcld_in_cm, Mcld_in_g, muu, Mach, grav_const_in_cgs, G]

try:
	os.remove('params.GPU')
except:
	pass

outF = open('params.GPU', "a")
for i in range(len(paramz)):
	outF.write(str(paramz[i]))
	outF.write('\n')
outF.close()

#************** END of GPU IC Creation *********



#***********************************************
#**************** FOR GADGET *******************
#***********************************************
UnitMass_in_g = Mcld_in_g
UnitRadius_in_cm = Rcld_in_cm

masses = m.copy()
unitVelocity = (grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm)**0.5
u = np.zeros(len(masses)) + c_0**2 / unitVelocity**2

#rhox = np.concatenate((rho, rho))
entr = (gamma - 1.0) * u / rho**(gamma - 1.0)

print('********************************************')
print('******* To be Used in Gadget EOS ***********')
print('********************************************')
print('UnitMass_in_g = ', UnitMass_in_g)
print('UnitRadius_in_cm = ', UnitRadius_in_cm)
print('unitVelocity = ', unitVelocity)
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3
print('UnitDensity_in_cgs = ', UnitDensity_in_cgs)
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
Unit_P_in_cgs = UnitDensity_in_cgs * Unit_u_in_cgs
print('Unit_P_in_cgs = ', Unit_P_in_cgs)
unitTime = (UnitRadius_in_cm**3/grav_const_in_cgs/UnitMass_in_g)**0.5
unitTime_in_Myr = unitTime / 3600. / 24. / 365.25 / 1.e6
print('unitTime_in_Myr = ', unitTime_in_Myr)
print('********************************************')
print()
print('np.sort(rho) = ', np.sort(rho) * UnitDensity_in_cgs)


vel /= unitVelocity

pos = r.copy()

ids = np.arange(pos.shape[0])

FloatType = np.float32  # double precision: np.float64, for single use np.float32
IntType = np.int32

IC = h5py.File('hfv_' + num + 'k_ic.hdf5', 'w')

## create hdf5 groups
header = IC.create_group("Header")
part0 = IC.create_group("PartType0")

## header entries
NumPart = np.array([pos.shape[0]], dtype=IntType)
header.attrs.create("NumPart_ThisFile", NumPart)
header.attrs.create("NumPart_Total", NumPart)
header.attrs.create("NumPart_Total_HighWord", np.zeros(1, dtype=IntType) )
header.attrs.create("MassTable", np.zeros(1, dtype=IntType) )
header.attrs.create("Time", 0.0)
header.attrs.create("Redshift", 0.0)
header.attrs.create("BoxSize", 0)
header.attrs.create("NumFilesPerSnapshot", 1)
header.attrs.create("Omega0", 0.0)
header.attrs.create("OmegaB", 0.0)
header.attrs.create("OmegaLambda", 0.0)
header.attrs.create("HubbleParam", 1.0)
header.attrs.create("Flag_Sfr", 0)
header.attrs.create("Flag_Cooling", 0)
header.attrs.create("Flag_StellarAge", 0)
header.attrs.create("Flag_Metals", 0)
header.attrs.create("Flag_Feedback", 0)
if pos.dtype == np.float64:
    header.attrs.create("Flag_DoublePrecision", 1)
else:
    header.attrs.create("Flag_DoublePrecision", 0)

## copy datasets
part0.create_dataset("Coordinates", data=r)
part0.create_dataset("Velocities", data=vel)
part0.create_dataset("ParticleIDs", data=ids )
part0.create_dataset("Masses", data=masses)
part0.create_dataset("InternalEnergy", data=entr)

IC.close()
#***********************************************

print()
print(r.shape)
print()
print('Done !')




