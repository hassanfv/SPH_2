
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants for conversion
R_inf_eV = 13.605693009  # Rydberg constant in eV
eV_to_J = 1.602176634e-19  # Conversion factor from eV to J
c = 2.99792458e8  # Speed of light in m/s
h = 6.62607015e-34  # Planck's constant in m^2 kg / s

# Load the data from the file
file_path = 'tableAGN.txt'
data = pd.read_csv(file_path, delim_whitespace=True, usecols=[0, 1], names=['E_ryd', 'flx'], comment='#')

E_ryd = data['E_ryd'].values
flx = data['flx'] # This is 4*pi*nu*Jnu (erg/s/cm^2) ----> For CHIMES should be converted to Jnu (erg/s/cm^2/sr/Hz)
nu = E_ryd * R_inf_eV * eV_to_J * c / h
Jnu = flx / 4.0 / np.pi / nu

plt.scatter(np.log10(E_ryd), np.log10(Jnu), s = 10, label = 'tableAGN from CLOUDY')

#-----------
df = pd.read_csv('B87_from_chimes.csv')
E = df['E_ryd'].values
Jnu = df['J_nu'].values
#-----------

#-----------
dfx = pd.read_csv('AGNtable6.3.csv')
rydx = dfx['Ryd'].values
logFnu = dfx['logFnu'].values
Fnu = 10**logFnu
Jnux = Fnu / 4.0 / np.pi
plt.plot(np.log10(rydx), np.log10(Jnux), color = 'red', label = 'AGN from Table 6.3 in Hazy 1')
#-----------

plt.scatter(E, Jnu, s = 10, label = 'B87 from CHIMES')

plt.legend()

plt.savefig('sed.png')

plt.show()

# Save the extracted columns to a new CSV file
#output_file_path = 'extracted_columns.csv'
#data.to_csv(output_file_path, index=False)

