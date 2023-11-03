
import numpy as np
import matplotlib.pyplot as plt

#filename = 'G-0.001313.bin' # No cooling
#filename = './OutAGN_1.0kpc_New/G-0.000300.bin' # With cooling

#filename = './OutAGN_1.0kpc_res_by_2/G-0.004450.bin'

filename = 'G-0.000088.bin'


import struct

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

ionFrac = ionFrac.reshape((N, 14))

print('Typ == 0 ===> ', np.sum(Typ == 0))

print('ionFrac.shape = ', ionFrac.shape)

n = np.where(u != 0.0)[0]
rho = rho[n]
u = u[n]
h = h[n]


#plt.scatter(rho, h, s = 0.1)
#plt.show()


#plt.hist(h, bins = 20)
#plt.show()

#print(h[1339855:])
# 1339855



ionFrac = ionFrac[n, :]

x = x[n]
y = y[n]
z = z[n]

vx = vx[n]
vy = vy[n]
vz = vz[n]

nz = np.where(np.abs(y) < 0.04)[0]

x = x[nz]
y = y[nz]
z = z[nz]

vx = vx[nz]
vy = vy[nz]
vz = vz[nz]

h = h[nz]


ionFrac = ionFrac[nz, :]

#====== radial velocity plot =====

unit_velocity_cgs = 3.086e+21/1.16462e+15 # cm/s

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

plt.scatter(r, vr, s = 5, color = 'k')
#plt.scatter(r, ur, s = 5, color = 'k')
plt.show()



#==================================

u = u[nz]
rho = rho[nz]

nx = np.where(rho == max(rho))[0]
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

unit_u = 7.02136e+12
gamma = 5./3.
Temp = (gamma - 1) * mH / kB * mu * u * unit_u
print('sort T = ', np.sort(Temp))#[-5:])

print('median(Temp) = ', np.median(Temp))


unit_rho = 1.10465e-23 # !!!!!!!!!!!!!!!!!!!
rho_cgs = rho * unit_rho
XH = 0.7
nH_cgs = rho_cgs * XH / mH


plt.hist(np.log10(Temp), bins = 50)
plt.show()

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

print(f'max(nH) = {max(rho * unit_density_in_cgs * XH /mH)}')
print()

plt.hist(nH, bins = 50)
plt.title('nH')
plt.ylabel('N (Frequency)')
plt.yscale('log')
plt.show()

print('rho[nn] = ', rho[nn]*unit_density_in_cgs)
XH = 0.7
print('nH[nn] = ', rho[nn]*unit_density_in_cgs * XH /mH)


plt.scatter(np.log10(nH), np.log10(Temp), s = 0.1, color = 'k')
plt.show()

print()
print('sort(Temp) = ', np.sort(Temp))
print()

nk = np.where(nH > 1500)[0]
xT = x[nk]
yT = y[nk]
zT = z[nk]
TempT = Temp[nk]
print('nk = ', nk)
print()
print('Temp[nk] = ', Temp[nk])
print()

plt.figure(figsize=(10, 8))

# Create a scatter plot. The color of each point will depend on the corresponding T value.
scatter = plt.scatter(x, z, c=np.log10(Temp), cmap='rainbow', s=20)
#scatter = plt.scatter(x, y, c=np.log10(nH_cgs), cmap='rainbow', s=2)




# Add a colorbar to the plot to show the relationship between color and T value.
#plt.colorbar(scatter, label='T Value')
plt.colorbar(scatter, label='nH Value')

#----- drawing a circle ----
R = 0.1
center = (0, 0)
theta = np.linspace(0, 2*np.pi, 100)
xc = center[0] + R * np.cos(theta)
yc = center[1] + R * np.sin(theta)

plt.plot(xc, yc, color = 'orange', linewidth = 5)
#---------------------------

xy = 0.56

plt.xlim(-xy, xy)
plt.ylim(-xy, xy)


plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of X and Y, colored by T value')

plt.savefig('fig.png')

plt.show()




