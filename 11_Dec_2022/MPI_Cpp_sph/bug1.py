
import numpy as np

N_tot = 11000

with open('IC_000k.bin', "rb") as file:
    Typ_read = np.fromfile(file, dtype=np.int32, count=N_tot)
    x_read = np.fromfile(file, dtype=np.float32, count=N_tot)
    y_read = np.fromfile(file, dtype=np.float32, count=N_tot)
    z_read = np.fromfile(file, dtype=np.float32, count=N_tot)
    vx_read = np.fromfile(file, dtype=np.float32, count=N_tot)
    vy_read = np.fromfile(file, dtype=np.float32, count=N_tot)
    vz_read = np.fromfile(file, dtype=np.float32, count=N_tot)
    mass_read = np.fromfile(file, dtype=np.float32, count=N_tot)
    h_read = np.fromfile(file, dtype=np.float32, count=N_tot)
    epsilon_read = np.fromfile(file, dtype=np.float32, count=N_tot)
    u_read = np.fromfile(file, dtype=np.float32, count=N_tot)


print(x_read[:10])
print(y_read[:10])
print(z_read[:10])
print(u_read)
