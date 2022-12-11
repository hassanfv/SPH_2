
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time
from libsx import *


T111 = time.time()

M_sun = 1.989e33 # gram
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2
UnitMass_in_g = 50.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
UnitRadius_in_cm = 0.8 * 3.086e18 # cm (2.0 pc)    #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
Unit_P_in_cgs = UnitDensity_in_cgs * Unit_u_in_cgs
unitVelocity = (grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm)**0.5

unitTime = (UnitRadius_in_cm**3/grav_const_in_cgs/UnitMass_in_g)**0.5
unitTime_in_yr = unitTime / 3600. / 24. / 365.25
unitTime_in_Myr = unitTime / 3600. / 24. / 365.25 / 1e6

print('unitTime_in_Myr = ', unitTime_in_Myr)
print('unitTime_in_yr = ', unitTime_in_yr)


mH = 1.6726e-24 # gram
mH2 = 2.7 * mH


filz = np.sort(glob.glob('./Outputs/*.pkl'))
filz = np.sort(glob.glob('/mnt/Linux_Shared_Folder_2022/AWS_16_April_Done/Outputs/*.pkl'))

j = 300

with open(filz[j], 'rb') as f:
	data = pickle.load(f)


r = data['pos']
t = data['current_t']


Th1 = time.time()
#----------- h  -------------
h = do_smoothingX((r, r))  # This plays the role of the initial h so that the code can start !
#----------------------------
print('Th1 = ', time.time() - Th1)

#-------- rho ---------
Trho = time.time()
rho = getDensity(r, m, h) * UnitDensity_in_cgs # g/cm^3
print('Trho = ', time.time() - Trho)
#----------------------

nH = rho / mH

print()
print('Current time in Myr = ', t * unitTime_in_Myr)
print('nH = ', np.sort(nH))

print()
print('Total Elapsed time = ', time.time() - T111)







