
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#/* ryd_to_eV */
def ryd_to_eV(ryd):
  return 13.60 * ryd


#/* eV_to_neu */
def eV_to_neu(eV):
  h_in_eV = 4.135667696e-15 # /* h in eV.Hz^-1 */
  return eV / h_in_eV;


#/* neu_to_lamb (lamb will be in Angstrom) */
def neu_to_lamb(neu):
  clight = 29979245800. # /* cm/s */
  return clight / neu * 1e8


#/* eV_to_lamb (lamb will be in Angstrom)*/
def eV_to_lamb(eV):
  neu = eV_to_neu(eV)
  return neu_to_lamb(neu)

#/* ryd_to_lamb*/
def ryd_to_lamb(ryd):
  eV = ryd_to_eV(ryd)
  return eV_to_lamb(eV)


#==== closestndx
def closestndx(arr, x):
  diff = np.abs(arr - x)
  return np.argmin(diff)
  


df = pd.read_csv('Selsing_composite.csv')

wav = df['wav'].values
flx = df['flx'].values
err = df['err'].values

alpha = 1.70

lm0 = 1468. # A
F_lm0 = 12.57

C = F_lm0 / lm0**(-alpha)

wgrid = np.arange(45.0, max(wav), 0.1)

ref_wav = 912. # A
nx = closestndx(wgrid, ref_wav)
print('nx =', nx)

L_at_ref_wav = 1.1e42 # erg/s/A

cont = C * wgrid ** (-alpha)

#plt.plot(wav, flx, color = 'k')
#plt.plot(wgrid, cont, color = 'r')
#plt.show()


L = cont / cont[nx] * L_at_ref_wav

dw = wgrid[1] - wgrid[0]

s = 0.0

for i in range(len(L)):

  s += L[i] * dw


print(f'L_bol = {np.log10(s):.3f}')

print(f'20 ryd corresponds to {ryd_to_lamb(20.0)} Angstrom')

plt.plot(wgrid, np.log10(L), color = 'k')
plt.show()






