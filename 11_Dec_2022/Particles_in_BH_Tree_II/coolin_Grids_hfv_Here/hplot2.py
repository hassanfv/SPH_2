
import matplotlib.pyplot as plt 
import sys
import os
import h5py
import numpy as np
import pandas as pd


#===== list_all_items
def list_all_items(h5file):
    def recurse(name, node):
        if isinstance(node, h5py.Dataset):
            print(f'Dataset: {name}')
        elif isinstance(node, h5py.Group):
            print(f'Group: {name}')
            for sub_name in node:
                recurse(f'{name}/{sub_name}', node[sub_name])

    for item in h5file:
        recurse(item, h5file[item])




## Read in equilibrium table
h5file = h5py.File('cooling_rates.hdf5', "r")

list_all_items(h5file)

s()

df = {"temperature" : None,
                    "density" : None,
                    "metallicity" : None} 

T = np.array(h5file["TableBins/Temperatures"]) 
nH = np.array(h5file["TableBins/Densities"]) 
Z = np.array(h5file["TableBins/Metallicities"]) 

# Read in the cooling and heating rates, 
# in log10(erg/cm3/s) 
log_cooling_rate = np.array(h5file["log_cooling_rate"]) 
log_heating_rate = np.array(h5file["log_heating_rate"]) 

print(log_cooling_rate.shape)

print(T)

i_nH = 7
i_Z = 1

cool = log_cooling_rate[:, i_nH, i_Z]

nHtmp = 10**nH[i_nH]

cool = [np.log10(10**tmp/nHtmp/nHtmp) for tmp in cool]


#----- From cloudy ---
df = pd.read_csv('cloudyHCool.csv')
# Temperature,HeatingRate,CoolingRate
T_cldy = df['Temperature']
Cool_cldy = df['CoolingRate']
Heat_cldy = df['HeatingRate']

plt.plot(T, cool, color = 'k')
plt.plot(T_cldy, Cool_cldy, color = 'b')

plt.show()

h5file.close()





