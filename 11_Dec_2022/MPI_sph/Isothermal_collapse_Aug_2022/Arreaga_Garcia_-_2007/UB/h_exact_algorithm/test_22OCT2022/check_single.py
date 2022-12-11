
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time


#unitTime_in_Myr =  0.07673840095663824 # Myr

M_sun = 1.98992e+33 # gram
UnitMass_in_g = 1.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
UnitRadius_in_cm = 4.99e+16  #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3

print(f'UnitDensity_in_cgs = {UnitDensity_in_cgs} g/cm^3')




#filz = np.sort(glob.glob('./Outputs/*.pkl'))

#filz = np.sort(glob.glob('/mnt/Linux_Shared_Folder_2022/Outputs_63k/*.pkl'))


#plt.ion()
fig, ax = plt.subplots(figsize = (6, 5))

#kb = ''

with open('00260.pkl', 'rb') as f:     # 62.23 + 4 = 66.23
	data = pickle.load(f)


r = data['pos']
h = data['h']
print(r.shape)

print('h = ', np.sort(h))

x = r[:, 0]
y = r[:, 1]
z = r[:, 2]
t = data['current_t']
rho = data['rho']
unitTime_in_kyr = data['unitTime']/3600./24./365.25/1000.

rho = np.sort(rho)*UnitDensity_in_cgs
print('rho = ', rho)

nrho = np.sum(rho >= 1e-15)

print('N_core = ', nrho) # See Commercon et al (2008).

ax.cla()

ax.scatter(x, y, s = 0.01, color = 'black')
xyrange = 1.5

ax.axis(xmin = -xyrange, xmax = xyrange)
ax.axis(ymin = -xyrange, ymax = xyrange)


ax.set_title('t = ' + str(np.round(t*unitTime_in_kyr,2)) + '       t/t_ff = ' + str(round(t*data['unitTime']/data['t_ff'], 4)))

plt.savefig('1111.png')

plt.show()








