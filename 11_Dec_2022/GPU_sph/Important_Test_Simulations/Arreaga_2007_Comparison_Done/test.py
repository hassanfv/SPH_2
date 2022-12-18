
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


with open('rho_vs_t_Gad.pkl', 'rb') as f:
	dfGad = pickle.load(f)

rho_Gad = dfGad['rho']
t_Gad = dfGad['t']


with open('rho_vs_t_hfv.pkl', 'rb') as f:
	dfhfv = pickle.load(f)

rho_hfv = dfhfv['rho']
t_hfv = dfhfv['t']


plt.scatter(t_hfv, np.log10(rho_hfv), s = 50, color = 'black', label = 'hfvSPH')
plt.scatter(t_Gad, np.log10(rho_Gad), s = 20, color = 'orange', label = 'Gadget')
plt.xlabel('t in code unit')
plt.ylabel('density in g/cm^3')
plt.legend()

plt.savefig('rho_vs_t_hfv_and_Gadget.png')

plt.show()


