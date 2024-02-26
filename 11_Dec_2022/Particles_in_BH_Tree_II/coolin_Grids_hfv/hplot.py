
import matplotlib.pyplot as plt 
import sys
import os
import h5py
import numpy as np 

## Read in equilibrium table
h5file = h5py.File('cooling_rates.hdf5', "r")

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

cool = log_cooling_rate[:, 7, 1]

plt.plot(T, cool, color = 'k')

plt.show()

h5file.close()





