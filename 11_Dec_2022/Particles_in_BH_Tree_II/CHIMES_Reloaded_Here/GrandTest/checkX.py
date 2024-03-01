
import numpy as np
import pickle



with open('float32_AbResX.bin', 'rb') as binary_file:
    # Read the entire file into a bytes object
    data_bytes = binary_file.read()

    # Convert the bytes object to a numpy array of type float32
    Ab = np.frombuffer(data_bytes, dtype=np.float32)

print(len(Ab))
print()

IDz = ['HI', 'HII', 'CI', 'CII', 'CIII', 'CIV', 'SiII', 'SiIII', 'SiIV', 'NV', 'OI' 'OVI', 'SII', 'FeII', 'MgI', 'MgII']

logT = np.arange(3, 11.1, 1.0)
lognH = np.arange(-2, 5.1, 1.0)
rkpc = np.arange(0.02, 0.33, 0.1)
logNHtot = np.arange(19, 23.1, .5)

N_T = len(logT)
N_nH = len(lognH)
N_r = len(rkpc)
N_NH = len(logNHtot)

N_t = 11 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DOUBLE CHECK !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

N_Elm = 15 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! You may Rarely Change it So be careful !!!!!!!!!!!!!!!!!!!!!!!

print(N_T * N_nH * N_r * N_NH * N_Elm * N_t)

S1 = N_nH * N_r * N_NH * N_Elm * N_t
S2 = N_r * N_NH * N_Elm * N_t
S3 = N_NH * N_Elm * N_t
S4 = N_Elm * N_t
S5 = N_t

i = 1 # T
j = 3 # nH
k = 3 # r
l = 2 # NH
m = 5 # Elm

print(f'Current params: T = {logT[i]}, nH = {lognH[j]}, r = {rkpc[k]}, NH = {logNHtot[l]}, Elm = {IDz[m]}, Lsh = {((10**logNHtot[l])/(10**lognH[j])):.3E}')

for t in range(N_t):
  
  Lnx = i * S1 + j * S2 + k * S3 + l * S4 + m * S5 + t
  
  print(Ab[Lnx])
  


