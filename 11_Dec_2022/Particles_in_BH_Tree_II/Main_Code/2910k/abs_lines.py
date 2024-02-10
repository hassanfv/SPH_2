
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d

#===== getTau
def getTau(lm, l0, f, N, b, gam):

  b = b * 1000.0 * 100.0 # cm/s
  c = 2.99792e10        # cm/s
  m_e = 9.1094e-28       # g
  e = 4.8032e-10        # cgs units

  a = l0 * 1e-8 * gam / 4.0 / np.pi / b

  K_i = e**2 * np.sqrt(np.pi) * f * l0 * 1e-8 / m_e / c / b

  lD = (b/c) * l0
  x = (lm - l0) / lD
  
  P = x**2
  H0 = np.exp(-x**2)
  Q = 1.5/x**2
  H_a_x = H0 - a/np.sqrt(np.pi)/P * (H0*H0*(4.*P*P + 7.*P + 4. + Q) - Q - 1)

  tau = K_i * 10**N * H_a_x

  return tau # np.exp(-tau) gives the normalized absorption profile.



#===== getTauDoublet
def getTauDoublet(lm, N, b, l0_a, f_a, gam_a, l0_b, f_b, gam_b):

  tau1 = getTau(lm, l0_a, f_a, N, b, gam_a)
  tau2 = getTau(lm, l0_b, f_b, N, b, gam_b)

  return tau1 + tau2 # np.exp(-tau) gives the normalized absorption profile.




df = pd.read_csv('atom.csv')
print(df)
N = df.shape[0]

iD = df['id'].values
lamb = df['lambda'].values
fosc = df['fosc'].values
gam = df['gam'].values

dfcols = pd.read_csv('columnDensities.csv')
MiD = dfcols['id'].values
Ncol = dfcols['Ncol'].values
b = dfcols['b'].values
mask = dfcols['mask'].values # 0 ---> do not make a plot. 1 ---> make a plot.
DB = dfcols['DB'].values # For doublets, the value of DB ONLY for the first line is set to 1

#----> Resolution <------
lambda0 = 2000.0 # a representative value
R_SDSS = 2000
FWHM = lambda0 / R_SDSS
sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))

k = 1

plt.figure(figsize = (10, 10))

for i in range(N):
  
  if mask[i] == 1:
    w1 = lamb[i] - 5.0
    w2 = lamb[i] + 5.0
    if DB[i] == 1:
      w1 = lamb[i] - 5.0
      w2 = lamb[i+1] + 5.0

    wgrid = np.linspace(w1, w2, 100)
    flx = np.zeros_like(wgrid)
    
    for j, w in enumerate(wgrid):
      tau = getTau(w, lamb[i], fosc[i], Ncol[i], b[i], gam[i])
      if DB[i] == 1:
        tau += getTau(w, lamb[i+1], fosc[i+1], Ncol[i+1], b[i+1], gam[i+1])
      flx[j] = np.exp(-tau)

    sigma_in_pixels = sigma / (wgrid[1] - wgrid[0])
    
    # degrading to SDSS resolution
    flx = gaussian_filter1d(flx, sigma_in_pixels)

    
    plt.subplot(4, 4, k)
    plt.plot(wgrid, flx, color = 'k')
    plt.text(w1+1, 0.15, iD[i], fontsize = 12, color = 'blue')
    plt.ylim(-0.05, 1.05)
    k += 1

    #l0_a, f_a, gam_a, l0_b, f_b, gam_b = 1548.204,0.189900,2.642E8, 1550.77755,0.094750,2.628E8
    #tau_CIV = getTauDoublet(lm, N, b, l0_a, f_a, gam_a, l0_b, f_b, gam_b)

plt.savefig('Fig.png')

plt.show()





