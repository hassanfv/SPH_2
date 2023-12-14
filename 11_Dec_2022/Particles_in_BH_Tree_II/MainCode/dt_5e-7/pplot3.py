
import numpy as np
import matplotlib.pyplot as plt
import struct

#filename = './No_T_limit/G-0.002200.bin'

filename = 'G-0.022000.bin'


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
        ngbDebug = np.array(struct.unpack(f'{N}i', f.read(N * 4)))  # Reading ngbDebug

    # Return the data
    return N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac, ngbDebug

# Usage
N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac, ngbDebug = readBinaryFile(filename)


print('Typ == 0 ===> ', np.sum(Typ == 0))

print('ionFrac.shape = ', ionFrac.shape)

#print(ionFrac[:14])


ntmp = np.where(h > 0.03)[0]
print('ntmp = ', ntmp)



n = np.where(u != 0.0)[0]
rho = rho[n]
u = u[n]

h = h[n]

print(np.sort(h))


#plt.hist(h, bins = 20)
#plt.show()



x = x[n]
y = y[n]
z = z[n]


vx = vx[n]
vy = vy[n]
vz = vz[n]

nz = np.where(np.abs(z) < 0.04)[0]

x = x[nz]
y = y[nz]
z = z[nz]

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

unit_velocity_cgs = 1.67601e+06 # cm/s

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

unit_u = 2.80901e+12 #!!!!!!!!!!!!!!!!!!!!!!!!
gamma = 5./3.
Temp = (gamma - 1) * mH / kB * mu * u * unit_u
print('sort T = ', np.sort(Temp))#[-5:])

print('median(Temp) = ', np.median(Temp))


unit_rho = 4.41933e-24 # !!!!!!!!!!!!!!!!!!!
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


ntmp = np.where(h > 0.03)[0]
print('ntmp = ', ntmp)


plt.figure(figsize=(10, 8))

# Create a scatter plot. The color of each point will depend on the corresponding T value.
scatter = plt.scatter(x, y, c=np.log10(Temp), cmap='rainbow', s=2)
#scatter = plt.scatter(x, y, c=np.log10(nH_cgs), cmap='rainbow', s=2)



#scatter = plt.scatter(x[ntmp], y[ntmp], c='lime', s=20)



# Add a colorbar to the plot to show the relationship between color and T value.
plt.colorbar(scatter, label='nH Value')

xy = 0.37

plt.xlim(-xy, xy)
plt.ylim(-xy, xy)


plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of X and Y, colored by T value')

plt.savefig('fig.png')

plt.show()




