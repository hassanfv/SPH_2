#!/usr/bin/env python3
"""
Code that plots different radial profiles for the Evrard collapse.
The results are compared with 1D PPM results from (Steinmetz & Mueller 1993) for t= 0.8

"""
# load libraries
import sys  # load sys; needed for exit codes
import numpy as np  # load numpy
import h5py  # load h5py; needed to read snapshots
import matplotlib
import matplotlib.pyplot as plt  ## needs to be active for plotting!
import csv
import pickle

matplotlib.rc_file_defaults()
FloatType = np.float64


#loads 1D PPM results
def load_ppm_result():
	gamma = 5./3.
	rost = 3./4./np.pi
	est = 1.054811e-1  / 1.05
	pst = rost*est
	vst = np.sqrt(est)
	rst = 1.257607
	time = 0

	radius = np.zeros(350)
	rho = np.zeros(350)
	vr = np.zeros(350)
	press = np.zeros(350)

	with open('./ppm_profile/ppm1oaf') as csvfile:
		readCSV = csv.reader(csvfile)
		line = 0
		for row in readCSV:
			line = line+1
			values = row[0].split()
			if(line == 1):
				time = values[1]
				continue
			if(line == 352):
				break

			radius[line -2] = float(values[1]) /rst*1e-11
			rho[line -2] = float(values[2]) /rost
			vr[line -2] = float(values[4]) /vst*1e-8
			press[line -2] = float(values[3])/pst*1e-16

	rho=rho*(3.0/(4*np.pi))
	press = press*(3.0/(4*np.pi))

	entropy = press / rho**gamma

	return radius, rho, vr, entropy


fig, ax = plt.subplots(1,3,figsize=(18,6))


radius_ppm, rho_ppm, vr_ppm, entropy_ppm = load_ppm_result()

ax[0].plot(radius_ppm, rho_ppm, color="k")
ax[1].plot(radius_ppm, vr_ppm, color="k")
ax[2].plot(radius_ppm, entropy_ppm, color="k")

ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[2].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlim(0.01,1)
ax[1].set_xlim(0.01,1)
ax[2].set_xlim(0.01,1)

ax[0].set_ylim(0.004,700)
ax[1].set_ylim(-2.0,2.0)
ax[2].set_ylim(0.0,0.2)

ax[0].set_xlabel("R")
ax[1].set_xlabel("R")
ax[2].set_xlabel("R")

ax[0].set_ylabel(r"$\rho$")
ax[1].set_ylabel(r"$V_R$")
ax[2].set_ylabel(r"$P/\rho^\gamma$")
plt.tight_layout()




plt.savefig("Evrardx.png")
plt.show()

