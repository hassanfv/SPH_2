
import time
import numpy as np
import random
import h5py
from libsx2_2t import *
import pickle
import pandas as pd

np.random.seed(42)

NSample = 300000 # The desired number of particles to be included in ONE cloud in the IC.
NSample = int(NSample)

with open('Main_IC_Grid_633k.pkl', 'rb') as f:
	data = pickle.load(f)

r = data['r']

if NSample > r.shape[0]:
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	print('WARNING !!! NSample is greater than r.shape !!! Please change NSample!!!!')
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	exit(0)

Ntot = r.shape[0]
nn = np.arange(Ntot)
np.random.shuffle(nn)
rows = nn[:NSample]

r = data['r'][rows, :]
v = data['v'][rows, :]
m = data['m'][rows]

m = m / np.sum(m) # Normalizing m! VERY IMPORTANT!!

h = data['h'][rows]
rho = data['rho'][rows]
u = data['u'][rows]

paramz = data['paramz']
paramz[0] = NSample # only NSample meeded to be updated!

#***********************************************
#**************** FOR GPU **********************
#***********************************************

x = np.round(r[:, 0], 5)
y = np.round(r[:, 1], 5)
z = np.round(r[:, 2], 5)

vx = np.round(v[:, 0], 5)
vy = np.round(v[:, 1], 5)
vz = np.round(v[:, 2], 5)

epsilon = 0.001 + np.zeros(len(x))

dictx = {'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz, 'm': m, 'h': np.round(h, 6), 'eps': epsilon}

df = pd.DataFrame(dictx)

num = str(int(np.floor(NSample/1000)))
df.to_csv('GPU_IC_Arrega_' + num + 'k.csv', index = False, header = False)

#-------
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
gamma = paramz[2]
entr = (gamma - 1.0) * u / rho**(gamma - 1.0)

print('********************************************')
print('******* To be Used in Gadget EOS ***********')
print('********************************************')
UnitMass_in_g = paramz[5]
print('UnitMass_in_g = ', UnitMass_in_g)
UnitRadius_in_cm = paramz[4]
print('UnitRadius_in_cm = ', UnitRadius_in_cm)
grav_const_in_cgs = paramz[8]
unitVelocity = (grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm)**0.5
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


#vel /= unitVelocity

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
part0.create_dataset("Velocities", data=v)
part0.create_dataset("ParticleIDs", data=ids )
part0.create_dataset("Masses", data=m)
part0.create_dataset("InternalEnergy", data=entr)

IC.close()
#***********************************************

print('m = ', np.sort(m))

print()
print(r.shape)
print()
print('Done !')






