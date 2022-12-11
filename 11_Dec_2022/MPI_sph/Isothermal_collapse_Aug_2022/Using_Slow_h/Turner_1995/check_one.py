
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time


#unitTime_in_Myr =  0.6251515693750652 # Myr
M_sun = 1.989e33 # gram
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2
Mcld = UnitMass_in_g = 225.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
R_0 = 1.81 # see the printed output of step_2_IC_Turner_1995.py
UnitRadius_in_cm = R_0 * 3.086e18  #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3


fig, ax = plt.subplots(figsize = (10, 6))

with open('00650.pkl', 'rb') as f:
	data = pickle.load(f)


r = data['pos']
h = data['h']

print('h = ', np.sort(h))

x = r[:, 0]
y = r[:, 1]
z = r[:, 2]
t = data['current_t']
rho = data['rho']
unitTime = data['unitTime']
unitTime_in_Myr = unitTime / 3600. / 24. / 365.25 / 1.e6
print('rho = ', np.sort(rho)*UnitDensity_in_cgs)

ax.scatter(x, y, s = 0.02, color = 'black')
#ax.scatter(y, z, s = 0.01, color = 'black')

xyrange = 0.04

#ax.axis(xmin = 1-xyrange, xmax = 1+xyrange)
#ax.axis(ymin = -xyrange, ymax = xyrange)

ax.axis(xmin = -1.2, xmax = 3.2)
ax.axis(ymin = -1.2, ymax = 1.5)



ax.set_title('t = ' + str(np.round(t*unitTime_in_Myr,4)))

plt.savefig('11_one.png')

plt.show()







