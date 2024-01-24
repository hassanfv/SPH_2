
import numpy as np
import matplotlib.pyplot as plt


T = np.logspace(4, 8)


Gam_e_H0 = 5.85e-11 * T**0.5 * np.exp(-157809.1/T) / (1.0 + (T/1e5)**0.5)

a_Hp = 8.4e-11 / T**0.5 / (T/1e3)**0.2 / (1.0 + (T/1e6)**0.7)







if False:
  plt.scatter(np.log10(T), np.log10(Gam_e_H0), label = 'Gam_e_H0')
  plt.scatter(np.log10(T), np.log10(a_Hp), label = 'a_Hp')
  plt.legend()
  plt.show()


