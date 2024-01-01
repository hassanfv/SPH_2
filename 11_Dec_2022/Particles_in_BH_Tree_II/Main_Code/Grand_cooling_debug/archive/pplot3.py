
import numpy as np
import matplotlib.pyplot as plt
import struct


#filename = './Outputs/G-0.002860.bin'

filename = './Outputs/G-0.001880.bin'

unit_velocity_cgs = 1.34181e+06 # cm/s #!!!!!!!!!!!!!!!!!!!!!!!!
unit_u = 1.80046e+12 #!!!!!!!!!!!!!!!!!!!!!!!!
unit_rho = 2.83261e-24 # !!!!!!!!!!!!!!!!!!!

def readBinaryFile(filename):
    with open(filename, 'rb') as f:
        file_content = f.read()

    # Create a memoryview from the file content
    buffer = memoryview(file_content)

    # Function to read a chunk of data and update the offset
    def read_chunk(dtype, count, offset):
        dtype_size = np.dtype(dtype).itemsize
        data = np.frombuffer(buffer, dtype=dtype, count=count, offset=offset)
        offset += dtype_size * count
        return data, offset

    # Initialize offset
    offset = 0

    # Read N
    N, offset = read_chunk(np.int32, 1, offset)
    N = N[0]  # Extract the single integer value

    # Read arrays
    Typ, offset = read_chunk(np.int32, N, offset)
    x, offset = read_chunk(np.float32, N, offset)
    y, offset = read_chunk(np.float32, N, offset)
    z, offset = read_chunk(np.float32, N, offset)
    vx, offset = read_chunk(np.float32, N, offset)
    vy, offset = read_chunk(np.float32, N, offset)
    vz, offset = read_chunk(np.float32, N, offset)
    rho, offset = read_chunk(np.float32, N, offset)
    h, offset = read_chunk(np.float32, N, offset)
    uB, offset = read_chunk(np.float32, N, offset)
    uA, offset = read_chunk(np.float32, N, offset)
    u, offset = read_chunk(np.float32, N, offset)
    mass, offset = read_chunk(np.float32, N, offset)

    return x, y, z, vx, vy, vz, rho, h, uB, uA, u, mass, Typ, N

# Usage
x, y, z, vx, vy, vz, rho, h, uB, uA, u, mass, Typ, N = readBinaryFile(filename)


print('Typ == 0 ===> ', np.sum(Typ == 0))


ntmp = np.where(h > 0.03)[0]
print('ntmp = ', ntmp)

print('uB = ',uB)
print('uA = ',uA)
print('u = ',u)

print('mass = ', np.sort(mass))
print('len(mass) = ', len(mass))
print(mass[981379], u[981379], Typ[981379], h[981379], rho[981379])
print(mass[981378], mass[981379], mass[981380])
print(np.unique(mass))
plt.hist(mass, bins = 20)
plt.show()

s()

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


gamma = 5./3.
Temp = (gamma - 1) * mH / kB * mu * u * unit_u
print('sort T = ', np.sort(Temp))#[-5:])

print('median(Temp) = ', np.median(Temp))


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


ntmp = np.where((Temp > 60000) & (Temp < 100000))[0]
#print('ntmp = ', ntmp)

print()
print('Number of particles with T > 1e6 K (Note that this is for the thing layer) = ', len(np.where(Temp > 1e6)[0]))
print()
print('Number of particles with T > 1e9 K (Note that this is for the thing layer) = ', len(np.where(Temp > 1e9)[0]))
print()

plt.figure(figsize=(10, 8))

# Create a scatter plot. The color of each point will depend on the corresponding T value.
scatter = plt.scatter(x, y, c=np.log10(Temp), cmap='rainbow', s=2)
#scatter = plt.scatter(x, y, c=np.log10(nH_cgs), cmap='rainbow', s=2)



# Add a colorbar to the plot to show the relationship between color and T value.
plt.colorbar(scatter, label='nH Value')

#scatter = plt.scatter(x[ntmp], y[ntmp], c='lime', s=2)

xy = 0.32

plt.xlim(-xy, xy)
plt.ylim(-xy, xy)


plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of X and Y, colored by T value')

plt.savefig('fig.png')

plt.show()




