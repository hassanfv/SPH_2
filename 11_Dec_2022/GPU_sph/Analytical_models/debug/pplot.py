
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#===== T_s
def T_s(E_s_th, R_s, rho_0, R_0, alpha):

	Ms = M_s(R_s, rho_0, R_0, alpha)
	Ts = 28./69. * E_s_th * mH / Ms / kB
	
	return Ts

#==== M_s
def M_s(R_s, rho_0, R_0, alpha):

	return 4.0 * np.pi * rho_0 * R_0**alpha * R_s**(3. - alpha) / (3. - alpha)




df = pd.read_csv('dataX.csv')

print(df)

R_s = df['R_s'].values
v_s = df['v_s'].values
E_b = df['E_b'].values
E_s_th = df['E_s_th'].values
t = df['t'].values

tMyr = t / 1e6 / 365.25 / 24 / 3600


nH = 10.
XH = 0.9
mH = 1.6726e-24
kB = 1.3807e-16
rho_0 = nH * mH / XH
alpha = 0.0 # uniform density!
R_0 = 100 * 3.086e+18 # cm

Ts = T_s(E_s_th, R_s, rho_0, R_0, alpha)

#--- Output cleaned Ts to a file along with t. -----
nT = np.where((Ts >= 9000.0) & (Ts <= 15000.0))[0]
print(Ts[nT])

Ts = Ts[:93799]
Ts[-1] = 10000
tMyr = tMyr[:93799]

tGrid = np.linspace(tMyr[-1], 1.0, 10000)
Ts_tmp = np.ones(len(tGrid)) * 10000.0

Ts = np.concatenate([Ts, Ts_tmp])
tMyr = np.concatenate([tMyr, tGrid])

# interpolation to new time grid
tnew = np.linspace(min(tMyr), max(tMyr), 10000)
TsNew = np.interp(tnew, tMyr, Ts)
tMyr = tnew[1:]
Ts = TsNew[1:]

df_T = pd.DataFrame({'tMyr': tMyr, 'Ts': Ts})
df_T.to_csv('Ts_vs_Time.csv', index = False) # We will use the time grid in this file to interpolate R_s, v_s, E_b to the same grid as this.

#----------------------------------------------------

if True:
	plt.plot(tMyr, Ts, color = 'k', linewidth = 3)
	#plt.xscale('log')
	plt.yscale('log')
	plt.xlim(0, 1)
	plt.ylim(1e3, 1e10)
	
	plt.xlabel('t [Myr]')
	plt.ylabel('T [K]')
	
	plt.axhline(y = 10000, linestyle = ':', color = 'b')
	
	plt.savefig('fig.png')
	plt.show()



