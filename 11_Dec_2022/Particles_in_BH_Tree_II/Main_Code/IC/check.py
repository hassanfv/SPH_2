import numpy as np
import matplotlib.pyplot as plt


def read_bin(filename, N_particles, N_blank):
    N_tot = N_particles + N_blank

    with open(filename, "rb") as file:
        # Read Type array
        Typ_bytes = file.read(N_tot * 4)  # int32 has 4 bytes
        Typ = np.frombuffer(Typ_bytes, dtype=np.int32)

        # Read position arrays
        x_bytes = file.read(N_tot * 4)  # float32 has 4 bytes
        x = np.frombuffer(x_bytes, dtype=np.float32)

        y_bytes = file.read(N_tot * 4)
        y = np.frombuffer(y_bytes, dtype=np.float32)

        z_bytes = file.read(N_tot * 4)
        z = np.frombuffer(z_bytes, dtype=np.float32)

        # Read velocity arrays
        vx_bytes = file.read(N_tot * 4)
        vx = np.frombuffer(vx_bytes, dtype=np.float32)

        vy_bytes = file.read(N_tot * 4)
        vy = np.frombuffer(vy_bytes, dtype=np.float32)

        vz_bytes = file.read(N_tot * 4)
        vz = np.frombuffer(vz_bytes, dtype=np.float32)

        # Read other properties
        mass_bytes = file.read(N_tot * 4)
        mass = np.frombuffer(mass_bytes, dtype=np.float32)

        h_bytes = file.read(N_tot * 4)
        h = np.frombuffer(h_bytes, dtype=np.float32)

        epsilon_bytes = file.read(N_tot * 4)
        epsilon = np.frombuffer(epsilon_bytes, dtype=np.float32)

        u_bytes = file.read(N_tot * 4)
        u = np.frombuffer(u_bytes, dtype=np.float32)

    return Typ, x, y, z, vx, vy, vz, mass, h, epsilon, u

# Example usage

N_particles = 700000
N_blank = 20000

filename = 'IC_R_720k.bin'  # Replace '10k' with appropriate number
Typ, x, y, z, vx, vy, vz, mass, h, epsilon, u = read_bin(filename, N_particles, N_blank)

nx = u != 0.0

h = h[nx]

print(np.sort(h))

plt.scatter(x, y, s = 0.1)

plt.show()




