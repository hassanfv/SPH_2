
import numpy as np
import matplotlib.pyplot as plt


T = np.logspace(4, 8, 100)

Te = 8.6173e-5 * T # 8.6173e-5 is the conversion factor from Kelvin to eV!

a_Hep_1 = 3.925e-13 * Te**(-0.6353) # Grassi et al - 2011

a_Hep_2 = 1.5e-10 * T**(-0.6353) # Katz, Weinberg


plt.scatter(np.log10(T), np.log10(a_Hep_1), color = 'k', s = 20, label = 'Grasi et al - 2011')
plt.scatter(np.log10(T), np.log10(a_Hep_2), color = 'b', s = 5, label = 'Katz, Weinberg')

plt.legend()

plt.show()

