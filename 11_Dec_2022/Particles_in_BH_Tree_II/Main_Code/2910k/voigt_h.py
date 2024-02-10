
import numpy as np
import matplotlib.pyplot as plt


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



l0 = 1215.67 # Angstrom
gam = 6.265e8
f = 0.416400

b = 30.0 # km/s

N = 21.0 # the log10 of column density

wav = np.linspace(1200, 1236, 1000)

flx = np.zeros_like(wav)

for i in range(len(wav)):
  tau = getTau(wav[i], l0, f, N, b, gam)
  flx[i] = np.exp(-tau)


plt.scatter(wav, flx, s = 1)
plt.show()





