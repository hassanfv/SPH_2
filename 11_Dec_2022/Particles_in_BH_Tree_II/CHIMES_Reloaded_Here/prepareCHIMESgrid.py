
import numpy as np
import pickle
import struct


with open('float32_uRes.bin', 'rb') as binary_file:
    # Read the entire file into a bytes object
    data_bytes = binary_file.read()

    # Convert the bytes object to a numpy array of type float32
    uarr = np.frombuffer(data_bytes, dtype=np.float32)

print(len(uarr))
print()


TGrid = np.arange(2, 11.1, 0.1)
nHGrid = np.arange(-4, 5.1, 0.1)
rGrid = np.arange(0.02, 0.33, 0.1)
NHGrid = np.arange(16, 23.1, 0.2)

dtGrid = np.arange(0, 11, 1.0) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DOUBLE CHECK !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


N_T = len(TGrid)
N_nH = len(nHGrid)
N_r = len(rGrid)
N_NH = len(NHGrid)

print('N_T = ', N_T)

N_t = len(dtGrid) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DOUBLE CHECK !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

print(N_T * N_nH * N_r * N_NH * N_t)


nHLowBound = min(nHGrid)
nHUpBound = max(nHGrid) 

rLowBound = min(rGrid)
rUpBound = max(rGrid)

NHLowBound = min(NHGrid) 
NHUpBound = max((NHGrid))

timeLowBound = min(dtGrid)
timeUpBound = max(dtGrid) #!!!!!!!!!!!!!!!!!!!!!!!!!!! DOUBLE CHECK !!!!!!!!!!!!!!!!!!!!

nHGrid = nHGrid.astype(np.float32)
rGrid = rGrid.astype(np.float32)
NHGrid = NHGrid.astype(np.float32)
dtGrid = dtGrid.astype(np.float32)

with open('HCoolChimes.bin', 'wb') as file:
    # Writing integers
    file.write(struct.pack('5i', N_T, N_nH, N_r, N_NH, N_t))
    
    # Writing floats
    file.write(struct.pack('8f', nHLowBound, nHUpBound, rLowBound, rUpBound, NHLowBound, NHUpBound, timeLowBound, timeUpBound))
    
    # Writing arrays
    uarr.tofile(file)
    nHGrid.tofile(file)
    rGrid.tofile(file)
    NHGrid.tofile(file)
    dtGrid.tofile(file)





