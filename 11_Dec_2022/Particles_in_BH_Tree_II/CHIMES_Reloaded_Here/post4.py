
import numpy as np
import pickle



with open('ch-AbRes.pkl', 'rb') as f:
  lst = pickle.load(f)


lst = [np.round(np.log10(_+1e-30), 3) for _ in lst]

with open('x-AbRes.pkl', 'wb') as f:
  pickle.dump(lst, f)

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




