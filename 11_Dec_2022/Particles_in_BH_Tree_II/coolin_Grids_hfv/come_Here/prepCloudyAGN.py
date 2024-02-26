
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#-----------
dfx = pd.read_csv('AGNtable6.3.csv')
rydx = dfx['Ryd'].values # Rydbergh
logFnu = dfx['logFnu'].values
Fnu = 10**logFnu
Jnux = Fnu / 4.0 / np.pi # erg/s/cm^2/sr/Hz
#-----------

linear_interp = interp1d(np.log10(rydx), np.log10(Jnux))
new_logRyd = np.linspace(min(np.log10(rydx)), max(np.log10(rydx)), 50)
new_logJnu = linear_interp(new_logRyd)

#------> save to a file <--------
dictx = {'logRyd': new_logRyd, 'logJnu': new_logJnu}
dfT = pd.DataFrame(dictx)

dfT.to_csv('ReadyAGN.csv', sep = ' ', index = False) # This can be used in CHIMES to generate_cross_section!


plt.plot(new_logRyd, new_logJnu, color = 'black', label = 'Interploation')

plt.scatter(np.log10(rydx), np.log10(Jnux), s = 10, color = 'red', label = 'AGN from Table 6.3 in Hazy 1')

plt.axvline(x = np.log10(1.0), linestyle = '--', color = 'blue')
plt.axvline(x = np.log10(2.0), linestyle = '--', color = 'blue')

plt.legend()

plt.savefig('sed.png')

plt.show()

# Save the extracted columns to a new CSV file
#output_file_path = 'extracted_columns.csv'
#data.to_csv(output_file_path, index=False)

