import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as colors

def read_arrays_from_binary(filename):
    # Read the binary file
    with open(filename, 'rb') as file:
        # Read N and NG from the file
        N = np.frombuffer(file.read(4), dtype=np.int32)[0]
        NG = np.frombuffer(file.read(4), dtype=np.int32)[0]

        # Read the arrays from the file
        x = np.frombuffer(file.read(N * 4), dtype=np.float32)
        y = np.frombuffer(file.read(N * 4), dtype=np.float32)
        z = np.frombuffer(file.read(N * 4), dtype=np.float32)
        vx = np.frombuffer(file.read(N * 4), dtype=np.float32)
        vy = np.frombuffer(file.read(N * 4), dtype=np.float32)
        vz = np.frombuffer(file.read(N * 4), dtype=np.float32)
        rho = np.frombuffer(file.read(NG * 4), dtype=np.float32)
        h = np.frombuffer(file.read(NG * 4), dtype=np.float32)
        u = np.frombuffer(file.read(NG * 4), dtype=np.float32)

    return x, y, z, vx, vy, vz, rho, h, u, N, NG

# Specify the input file name
filename = 'Outputs/G-0.002583.bin'

# Read the arrays from the binary file
x, y, z, vx, vy, vz, rho, h, u, N, NG = read_arrays_from_binary(filename)

x = np.array(x)[:NG]
y = np.array(y)[:NG]
z = np.array(z)[:NG]
masses = np.full(NG, 1.0 / NG)


x = x[:NG]
y = y[:NG]
z = z[:NG]

#xt = x[(x > -0.2) & (x < 0.2)]
#yt = y[(x > -0.2) & (x < 0.2)]
#zt = z[(x > -0.2) & (x < 0.2)]

#xtt = xt[(yt > -0.2) & (yt < 0.2)]
#ytt = yt[(yt > -0.2) & (yt < 0.2)]
#ztt = zt[(yt > -0.2) & (yt < 0.2)]

#xx = xtt[(ztt > -0.1) & (ztt < 0.1)]
#yy = ytt[(ztt > -0.1) & (ztt < 0.1)]
#zz = ztt[(ztt > -0.1) & (ztt < 0.1)]

#x = x[(y > -0.2) & (y < 0.2)]
#y = y[(y > -0.2) & (y < 0.2)]
#z = z[(y > -0.2) & (y < 0.2)]

#print(x.shape, y.shape, z.shape)

#nz = np.where(np.abs(z) <= 0.05)[0]
#x = x[nz]
#y = y[nz]
#z = z[nz]
#masses = masses[nz]



# Set the number of bins for the 2D histogram
num_bins = 10000

# Create a 2D histogram of particle positions on the x-y plane
hist, xedges, yedges = np.histogram2d(x, y, bins=num_bins)

# Transpose the histogram for plotting
hist = hist.T

# Set up the extent of the plot
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


# Calculate the logarithmic range for the color scale
vmin = hist[hist > 0].min()
vmax = hist.max()
# Define the logarithmic color scale
norm = colors.LogNorm(vmin=vmin, vmax=vmax)

# Create the density plot
#plt.imshow(hist, extent=extent, origin='lower', cmap='hot', norm=norm, interpolation='nearest')
plt.imshow(hist, extent=extent, origin='lower', cmap='hot', interpolation='nearest')
plt.colorbar(label='Density')
plt.xlabel('X')
plt.ylabel('Y')
xy = 0.1
plt.xlim(-xy, xy)
plt.ylim(-xy, xy)
plt.title('Particle Density on X-Y Plane')

# Show the plot
plt.show()














