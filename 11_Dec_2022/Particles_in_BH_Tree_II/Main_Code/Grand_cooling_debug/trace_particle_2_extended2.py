
import numpy as np
import matplotlib.pyplot as plt
import struct
import glob


filez = np.sort(glob.glob('./Outputs/*.bin'))

t = [np.float32(tmp[12:-4]) for tmp in filez]


unit_velocity_cgs = 1.34181e+06 # cm/s #!!!!!!!!!!!!!!!!!!!!!!!!
unit_u = 1.80046e+12 #!!!!!!!!!!!!!!!!!!!!!!!!
unit_rho = 2.83261e-24 # !!!!!!!!!!!!!!!!!!!
mH = 1.673534e-24



def readBinaryFile(filename):
    with open(filename, 'rb') as file:
        # Read the total number of elements (N)
        N = np.fromfile(file, dtype=np.int32, count=1)[0]

        # Read arrays in the order they were written
        Typ = np.fromfile(file, dtype=np.int32, count=N)
        x = np.fromfile(file, dtype=np.float32, count=N)
        y = np.fromfile(file, dtype=np.float32, count=N)
        z = np.fromfile(file, dtype=np.float32, count=N)
        vx = np.fromfile(file, dtype=np.float32, count=N)
        vy = np.fromfile(file, dtype=np.float32, count=N)
        vz = np.fromfile(file, dtype=np.float32, count=N)
        rho = np.fromfile(file, dtype=np.float32, count=N)
        h = np.fromfile(file, dtype=np.float32, count=N)
        uB = np.fromfile(file, dtype=np.float32, count=N)
        uA = np.fromfile(file, dtype=np.float32, count=N)
        u = np.fromfile(file, dtype=np.float32, count=N)
        dudt = np.fromfile(file, dtype=np.float32, count=N)
        mass = np.fromfile(file, dtype=np.float32, count=N)

    return x, y, z, vx, vy, vz, rho, h, uB, uA, u, dudt, mass, Typ, N



jj = 866864   #[866864 884793 941378 943429]

res = []

for i, filename in enumerate(filez):
  x, y, z, vx, vy, vz, rho, h, uB, uA, u, dudt, mass, Typ, N = readBinaryFile(filename)

  n = np.where(u != 0.0)[0]
  
  rho = rho[n]
  
  uB_ad = uB[n]
  uA_ad = uA[n]
  uB_hc = uA_ad
  uA_hc = u[n]

  u = uA_hc

  dt = 8e-8

  dudt_ad = (uA_ad - uB_ad) / dt
  dudt_hc = (uA_hc - uB_hc) / dt


  h = h[n]
  
  x = x[n]
  y = y[n]
  z = z[n]
  
  nn = np.where(np.abs(z) < 0.035)[0]

  vx = vx[n]
  vy = vy[n]
  vz = vz[n]
  
  vr = np.sqrt(vx*vx + vy*vy + vz*vz)

  kB = 1.3807e-16
  mu = 0.61
  mH = 1.673534e-24
  gamma = 5./3.
  Temp = (gamma - 1) * mH / kB * mu * u * unit_u

  res.append([t[i],np.log10(Temp[jj]),rho[jj],h[jj],vr[jj]*unit_velocity_cgs/100000, uB_ad[jj], uA_ad[jj], uB_hc[jj], uA_hc[jj],dudt_ad[jj],dudt_hc[jj],dudt[jj]])
  
  ntmp = np.where((Temp > 6000000) & (Temp < 6500000) & (x > 0.0) & (y > 0.0) & (np.abs(z) < 0.035))[0]
  print(t[i], uB_ad[jj], uA_ad[jj], uB_hc[jj], uA_hc[jj], dudt_ad[jj], dudt_hc[jj], dudt[jj])
  print('ntmp = ', ntmp)
  print()


res = np.array(res)

t = res[:, 0]
T = res[:, 1]
rho = res[:, 2]
h = res[:, 3]
vr= res[:, 4]

uB_ad = res[:, 5]
uA_ad = res[:, 6]
uB_hc = res[:, 7]
uA_hc = res[:, 8]

dudt_ad = res[:, 9]
dudt_hc = res[:, 10]
dudt = res[:, 11]

#uB_ad[jj], uA_ad[jj], uB_hc[jj], uA_hc[jj]

T_norm = (T - np.min(T)) / (np.max(T) - np.min(T))
rho_norm = (rho - np.min(rho)) / (np.max(rho) - np.min(rho))
h_norm = (h - np.min(h)) / (np.max(h) - np.min(h))
vr_norm = (vr - np.min(vr)) / (np.max(vr) - np.min(vr))

fig, axs = plt.subplots(2, 4, figsize=(19, 10))  # Adjust figsize as needed

# Scatter plot for each subplot
axs[0, 0].scatter(t, T, s=20)
axs[0, 0].set_title("Temperature")
axs[0, 0].set_xlabel("t")
axs[0, 0].set_ylabel("Log10(Temp)")

axs[0, 1].scatter(t, rho, s=20)
axs[0, 1].set_title("Density")
axs[0, 1].set_xlabel("t")
axs[0, 1].set_ylabel("rho")

axs[0, 2].scatter(t, h, s=20)
axs[0, 2].set_title("Scale Height")
axs[0, 2].set_xlabel("t")
axs[0, 2].set_ylabel("h")

axs[0, 3].plot(t, uB_ad, label='uB_ad', linestyle='-', color='blue')
axs[0, 3].plot(t, uA_ad, label='uA_ad', linestyle='--', color='blue')
axs[0, 3].plot(t, uB_hc, label='uB_hc', linestyle='-', color='green')
axs[0, 3].plot(t, uA_hc, label='uA_hc', linestyle='--', color='green')
axs[0, 3].set_title("Adiabatic and HCool Components")
axs[0, 3].set_xlabel("t")
axs[0, 3].set_ylabel("Values")
axs[0, 3].set_yscale('symlog')
axs[0, 3].legend()

axs[1, 0].scatter(t, vr, s=20)
axs[1, 0].set_title("Radial Velocity")
axs[1, 0].set_xlabel("t")
axs[1, 0].set_ylabel("vr")

# New scatter plot
axs[1, 1].scatter(x[nn], y[nn], s=0.01, color='k')
axs[1, 1].scatter(x[jj], y[jj], s=20, color='r')
axs[1, 1].set_xlim(0, 0.30)
axs[1, 1].set_ylim(0, 0.30)
axs[1, 1].set_title("Position Scatter")
axs[1, 1].set_xlabel("x")
axs[1, 1].set_ylabel("y")

# Overlay plot
axs[1, 2].plot(t, T_norm, linewidth = 3, label='Temp')
axs[1, 2].plot(t, rho_norm, linewidth = 3, label='Density')
axs[1, 2].plot(t, h_norm, linewidth = 3, label='h')
axs[1, 2].plot(t, vr_norm, linewidth = 3, label='Velocity')
axs[1, 2].set_title("Overlay of Normalized Quantities")
axs[1, 2].set_xlabel("t")
axs[1, 2].set_ylabel("Normalized Values")
axs[1, 2].legend()

axs[1, 3].plot(t, dudt_ad, linewidth = 3, label='dudt_ad', color='red')
axs[1, 3].plot(t, dudt, linewidth = 3, label='dudt_add_orig', color='blue')
axs[1, 3].scatter(t, dudt_hc, s = 4, label='dudt_hc', color='orange')
axs[1, 3].scatter(t, np.abs(dudt_hc), s = 4, label='abs(dudt_hc)', color='lime')
axs[1, 3].set_title("Rate of Change of Energy")
axs[1, 3].set_xlabel("t")
axs[1, 3].set_ylabel("dudt")
axs[1, 3].set_yscale('symlog')
axs[1, 3].legend()

# Adjust layout
plt.tight_layout()

plt.savefig('figure.png')

# Show plot
plt.show()







