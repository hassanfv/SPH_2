
import numpy as np
import pickle



with open('CHIMES_EC2_hfv.pkl', 'rb') as f:
  df = pickle.load(f)


# Immediately extract and discard other parts to free up memory
uRes = df.pop('uRes', None)
df = None  # Hint to the garbage collector to free up the memory

# Process and save uRes
with open('ch-uRes.pkl', 'wb') as f:
    pickle.dump(uRes, f)

#Tres = df['Tres']
#uRes = df['uRes']
#muRes = df['muRes']
#AbRes = df['AbRes']

'''
with open('ch-TRes.pkl', 'wb') as f:
  pickle.dump(Tres, f)

with open('ch-uRes.pkl', 'wb') as f:
  pickle.dump(uRes, f)

with open('ch-muRes.pkl', 'wb') as f:
  pickle.dump(muRes, f)
'''

#with open('ch-AbRes.pkl', 'wb') as f:
#  pickle.dump(AbRes, f)

#print(len(Tres))
if False:
  logT = np.arange(2, 11.1, 0.1)
  lognH = np.arange(-4, 5.1, 0.1)
  rkpc = np.arange(0.02, 0.33, 0.1)
  logNHtot = np.arange(16, 23.1, 0.2)

  N_T = len(logT)
  N_nH = len(lognH)
  N_r = len(rkpc)
  N_NH = len(logNHtot)

  N_t = 11

  S1 = N_nH * N_r * N_NH * N_t
  S2 = N_r * N_NH * N_t
  S3 = N_NH * N_t
  S4 = N_t

  for i in range(N_T):

    j = 0
    k = 0
    l = 0
    m = 0 # at time zero!!

    Lnx = i * S1 + j * S2 + k * S3 + l * S4 + m
    
    print(Tres[Lnx])




