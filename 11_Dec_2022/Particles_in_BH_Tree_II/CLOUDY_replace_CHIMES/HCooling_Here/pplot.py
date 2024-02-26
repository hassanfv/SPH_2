
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dfHC = pd.read_csv('HC.csv')


nH = 1000.0 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

cool = 10**dfHC['Cooling']/nH/nH
heat = 10**dfHC['Heating']/nH/nH

plt.scatter(dfHC['T'], np.log10(cool), s = 10, color = 'blue')
plt.scatter(dfHC['T'], np.log10(heat), s = 10, color = 'red')

plt.ylim(-27, -21)

plt.savefig('HC.png')

plt.show()



