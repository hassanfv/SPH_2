
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time


#unitTime_in_Myr =  0.6251515693750652 # Myr
M_sun = 1.989e33 # gram
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2
Mcld = UnitMass_in_g = 2000.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
R_0 = 4.54 # see the printed output of step_2_IC_Turner_1995.py
UnitRadius_in_cm = R_0 * 3.086e18  #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3

print('UnitDensity_in_cgs = ', UnitDensity_in_cgs)

fig, ax = plt.subplots(figsize = (7, 5))
#fig, ax = plt.subplots(figsize = (6, 5))

with open('./Outputs/00624.pkl', 'rb') as f:
#with open('./Outputs_35k_Extremely_Fast/00600.pkl', 'rb') as f:
	data = pickle.load(f)


r = data['pos']
h = data['h']
v = data['v']

#print('vx = ', np.sort(v[:, 0]))

print()
print('h = ', np.sort(h))

x = r[:, 0]
y = r[:, 1]
z = r[:, 2]
t = data['current_t']
rho = data['rho']
unitTime = data['unitTime']
unitTime_in_Myr = unitTime / 3600. / 24. / 365.25 / 1.e6
print('unitTime_in_Myr = ', unitTime_in_Myr)
print()
print('rho = ', np.sort(rho)*UnitDensity_in_cgs)
print()
print('rho code = ', np.sort(rho))

#rho_h_3 = np.sort(rho*h**3)
#print('rho_h_3 = ', rho_h_3)

ax.scatter(x, y, s = 0.01, color = 'black')
#ax.scatter(y, z, s = 0.01, color = 'black')

xyrange = 1.0

#ax.axis(xmin = -xyrange, xmax = xyrange)
#ax.axis(ymin = -xyrange, ymax = xyrange)

ax.axis(xmin = -1.2, xmax = 3.2)
ax.axis(ymin = -1.5, ymax = 1.5)


ax.set_title('t_code = ' + str(np.round(t,4)) + '       t = ' + str(np.round(t*unitTime_in_Myr,4)))

plt.savefig('11_one.png')

plt.show()







