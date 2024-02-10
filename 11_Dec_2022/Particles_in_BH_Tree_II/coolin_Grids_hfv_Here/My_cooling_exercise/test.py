
import numpy as np
import matplotlib.pyplot as plt


T = np.logspace(4, 8, 100)

Te = 8.6173e-5 * T # 8.6173e-5 is the conversion factor from Kelvin to eV!

a_d_1 = 1.544e-9 * Te**(-1.5) * np.exp(-48.596/Te) * (0.3 + np.exp(-8.1/Te)) # Grassi et al - 2011

a_d_2 = 1.9e-3 * T**(-1.5) * np.exp(-470000./T) * (1.0 + 0.3 * np.exp(-94000/T)) # Katz, Weinberg


plt.scatter(np.log10(T), np.log10(a_d_1), color = 'k', s = 5, label = 'Grasi et al - 2011')
plt.scatter(np.log10(T), np.log10(a_d_2), color = 'b', s = 10, label = 'Katz, Weinberg')

plt.legend()

plt.show()

