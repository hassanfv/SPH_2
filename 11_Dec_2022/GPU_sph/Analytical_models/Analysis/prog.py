
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#==== M_s
def M_s(R_s, rho_0, R_0, alpha):

	return 4.0 * np.pi * rho_0 * R_0**alpha * R_s**(3. - alpha) / (3. - alpha)


#===== Mdot_in
def Mdot_in(Tou_in, L_AGN, clight, v_in):
	
	return Tou_in * L_AGN / clight / v_in


nH = 1.0e1
XH = 0.9
mH = 1.6726e-24
kB = 1.3807e-16
rho_0 = nH * mH / XH
R_0 = 100 * 3.086e+18 # cm
alpha = 0.0

Tou_in = 1.0
L_AGN = 1e45 # erg/s
clight = 29979245800. # cm/s
v_in = 30000 * 1000. * 100 # cm/s


dfTs = pd.read_csv('Ts_vs_Time.csv')
tMyr_ref = dfTs['tMyr'].values
Ts = dfTs['Ts'].values

df = pd.read_csv('Data.csv')
tMyr = df['tMyr'].values
Rs = df['Rs'].values
vs = df['vs'].values
Eb = df['Eb'].values

#--- interplating to the same grid as tMyr_ref
Rs = np.interp(tMyr_ref, tMyr, Rs)
vs = np.interp(tMyr_ref, tMyr, vs)
Eb = np.interp(tMyr_ref, tMyr, Eb)

tMyr = tMyr_ref # !!!!!! Note that tMyr_ref is rename here !!

Ms = M_s(Rs, rho_0, R_0, alpha)
Es_th = 69./28. * Ts * Ms * kB / mH
Es_kin = 0.5 * Ms * vs * vs
Es_tot = Es_th + Es_kin

E_tot = Es_tot + Eb

MdotIn = Mdot_in(Tou_in, L_AGN, clight, v_in)
E_in = 0.5 * MdotIn * v_in * v_in * (tMyr * 1e6 * 365.25 * 24 * 3600)


plt.plot(tMyr, E_tot/E_in, linewidth = 3, color = 'k')

plt.ylim(0.1, 1.2)
plt.xlim(0.0, 1.0)

plt.xlabel('Time (Myr)')
plt.ylabel('T_tot/E_in')
plt.title('See Fig.2 in Richings et al - 2018')

plt.savefig('fig.png')
plt.show()












