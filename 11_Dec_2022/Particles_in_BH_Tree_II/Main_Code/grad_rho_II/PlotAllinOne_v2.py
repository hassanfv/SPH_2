
import numpy as np
import matplotlib.pyplot as plt
import struct

# Is used with outputs with CLOUDY h/cooling implementation (24 Feb 2024).

filename = './Outputs/G-0.000040.bin'

#filename = './Outputs_NoHCooling/G-0.002800.bin'

#filename = '/mnt/Linux_Shared_Folder_2022/Out5k28Jan_1/G-0.147522.bin'

tCode = np.float64(filename[-12:-4]) / 10.0 #!!!!!!!!!!!!!!!!! Double check if during the file-saving the multiplication by 10 is actually done!!!!!!!!!!


unit_velocity_cgs = 1.34181e+06 # cm/s #!!!!!!!!!!!!!!!!!!!!!!!!
unit_u = 1.80046e+12 #!!!!!!!!!!!!!!!!!!!!!!!!
unit_rho = 2.83261e-24 # !!!!!!!!!!!!!!!!!!!
unitTime_in_s = 2.29987e+15 #!!!!!!!!!!!!!!!!!!!!!!!!!

t_in_kyrs = tCode * unitTime_in_s / 3600./24./365.25/1000.

def readBinaryFile(filename):
    with open(filename, 'rb') as f:
        # Read N and N_ionFrac
        N, N_ionFrac = struct.unpack('ii', f.read(2 * 4))  # 4 bytes each for two integers

        # Read arrays
        Typ = np.array(struct.unpack(f'{N}i', f.read(N * 4)))
        x = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        y = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        z = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        vx = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        vy = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        vz = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        rho = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        h = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        u = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        mass = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        ionFrac = np.array(struct.unpack(f'{N_ionFrac}f', f.read(N_ionFrac * 4)))

    # Return the data
    return N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac 

# Usage
N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac = readBinaryFile(filename)

print('Typ == 0 ===> ', np.sum(Typ == 0))
print()

n = np.where(u != 0.0)[0]
rho = rho[n]
u = u[n]
mass = mass[n]

print('sort(mass) = ', np.sort(mass))

h = h[n]
print('sort(h) = ', np.sort(h))
print()
print('sort(rho) = ', np.sort(rho))
print()

x = x[n]
y = y[n]
z = z[n]


vx = vx[n]
vy = vy[n]
vz = vz[n]

#------ > check single particle properties <------
if False:
  jk = 2788854
  vv = (vx*vx+vy*vy+vz*vz)**0.5
  print()
  print('sort(vv) = ', np.sort(vv))
  print()
  print(x[jk], y[jk], z[jk], vv[jk], rho[jk], h[jk])
  print()
  s()

nz = np.where(np.abs(z) < 0.03)[0]

x = x[nz]
y = y[nz]
z = z[nz]

u = u[nz]
rho = rho[nz]

dist = np.sqrt(x*x + y*y + z*z)

Rad = 0.2
tolerence = 0.005

Nsphere = np.sum((dist > (Rad - tolerence)) & (dist < (Rad + tolerence)))

print()
print(f'particles on the spehere of a  surface with R = {Rad} is {Nsphere} out of total of {len(dist)} particles.')
print()


h = h[nz]

vx = vx[nz]
vy = vy[nz]
vz = vz[nz]

vv = np.sqrt(vx*vx + vy*vy + vz*vz)

print()
print('sorted(vv) = ', np.sort(vv))
print()


#====== radial velocity plot =====

vr = np.sqrt(vx * vx + vy * vy + vz * vz)
r = np.sqrt(x * x + y * y + z * z)

grid = np.linspace(0, max(r), 50)

res = []

for i in range(len(grid)-1):

  ng = np.where(((r > grid[i]) & (r <= grid[i+1])))[0]
  
  if len(ng) > 0:
    vvr = np.median(vr[ng])
    uur = np.median(u[ng])
    
    res.append([grid[i], vvr, uur])

res = np.array(res)

r = res[:, 0]
vr = res[:, 1] * unit_velocity_cgs / 100 / 1000 # km/s
ur = res[:, 2]


fig, axs = plt.subplots(2, 3, figsize=(20, 10))


# Plot 1: Radial Velocity
axs[0, 0].scatter(r, vr, s=5, color='k')
axs[0, 0].set_title('Radial Velocity')
axs[0, 0].set_xlabel('r')
axs[0, 0].set_ylabel('vr (km/s)')
#==================================

nx = np.where(u == max(u))[0]
print(f'median(u) = {np.median(u)}, max(u) = {u[nx]},  =====> nx = {nx}')

print('sort(u) = ', np.sort(u))

print(np.sum(Typ == -1))

nx = np.where(rho == max(rho))[0]
print(f'median(rho) = {np.median(rho)}, max(rho) = {rho[nx]}  NOTE: rho is different from nH ! rho here is in code unit !!!')
print(np.sort(rho))
#print(np.sum(rho >= 4.0))

kB = 1.3807e-16
mu = 0.61
mH = 1.673534e-24


gamma = 5./3.
Temp = (gamma - 1) * mH / kB * mu * u * unit_u
print('sort T = ', np.sort(Temp))#[-5:])

print('median(Temp) = ', np.median(Temp))


rho_cgs = rho * unit_rho
XH = 0.7
nH_cgs = rho_cgs * XH / mH

Tempx = Temp[Temp > 10**4.8]

# Plot 2: Temperature Histogram
axs[0, 1].hist(np.log10(Tempx), bins=50, density = True)
axs[0, 1].set_title('Temperature Histogram')
axs[0, 1].set_xlabel('log10(Temperature)')
axs[0, 1].set_ylabel('Frequency')

nT = np.where(Temp < 12000)[0]
print(nT)


#nn = np.where((x > 0.2) & ( Temp > 1e6))[0]  # nn =  [300013 300018 300022 300049 300104 300116 300163 300167 300178]
#print('nn = ', nn)
nn = 309

Temp_nn = (gamma - 1) * mH / kB * mu * u[nn] * unit_u
print()
print(f'Temp_nn = {Temp_nn} K')
print()

unit_density_in_cgs = unit_rho

nH = rho * unit_density_in_cgs * XH /mH

print(f'max(nH) = {max(rho * unit_density_in_cgs * XH /mH)} ---->  note this could be different from the global Max(nH).')
print()

# Plot 3: nH Histogram
axs[0, 2].hist(nH, bins=50)
axs[0, 2].set_title('nH Histogram')
axs[0, 2].set_xlabel('nH')
axs[0, 2].set_ylabel('Frequency')
axs[0, 2].set_yscale('log')

print('rho[nn] = ', rho[nn]*unit_density_in_cgs)
XH = 0.7
print('nH[nn] = ', rho[nn]*unit_density_in_cgs * XH /mH)

ntmp = np.where((Temp < 30000) & (nH > 0.01) & (nH < 5))[0]

# Plot 4: Scatter Plot (nH vs Temperature)
axs[1, 0].scatter(np.log10(nH), np.log10(Temp), s=0.01, color='k')
axs[1, 0].set_title('nH vs Temperature')
axs[1, 0].set_xlabel('log10(nH)')
axs[1, 0].set_ylabel('log10(Temp)')

print()
print('sort(Temp) = ', np.sort(Temp))
print()

nk = np.where((Temp > 1e6) & (Temp < 1e9))[0]
xT = x[nk]
yT = y[nk]
zT = z[nk]
TempT = Temp[nk]
#print('nk = ', nk)
print()
print(f'Number of particles with 1e6 < Temp < 1e9 is {len(TempT)}')
#print('Temp[nk] = ', Temp[nk])
print()

ntmp = np.where((Temp > 60000) & (Temp < 100000))[0]
#print('ntmp = ', ntmp)

print()
print('Number of particles with T > 1e6 K (Note that this is for the thing layer) = ', len(np.where(Temp > 1e6)[0]))
print()
print('Number of particles with T > 1e9 K (Note that this is for the thing layer) = ', len(np.where(Temp > 1e9)[0]))
print()

#plt.figure(figsize=(10, 8))

# Plot 5: Scatter Plot (X and Y, colored by Temperature)
scatter = axs[1, 1].scatter(x, y, c=np.log10(Temp), cmap='rainbow', s=0.2)
#scatter = axs[1, 1].scatter(x, y, c=np.log10(nH_cgs), cmap='rainbow', s=0.2)

fig.colorbar(scatter, ax=axs[1, 1], label='log10(Temperature)')

xy = 0.31

axs[1, 1].set_title(f'current time in kyears = {round(t_in_kyrs, 2)}')

axs[1, 1].set_xlim(-xy, xy)
axs[1, 1].set_ylim(-xy, xy)
#axs[1, 1].set_title('X and Y, colored by Temperature')
axs[1, 1].set_xlabel('X')
axs[1, 1].set_ylabel('Y')

# Optional: Adjust layout
plt.tight_layout()


plt.savefig('fig.png')

plt.show()




