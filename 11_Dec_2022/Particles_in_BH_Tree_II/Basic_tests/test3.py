
# Centered-differencing algorithom (section 3.3.1 from https://www.ita.uni-heidelberg.de/~dullemond/lectures/num_fluid_2011/Chapter_3.pdf)

import numpy as np
import matplotlib.pyplot as plt


dx = 1.0

x0 = 0.0
x1 = 100.0

Np = np.int32((x1 - x0) / dx)

Nt = 300 # We want to evolve the system for 300 time step!

q = np.zeros((Np, Nt)) # 1st column ===> x     2nd column ====> t

#--- defining q at t = t0 (initial condition value)
q[:30, 0] = 1.0
q[30:, 0] = 0.0
#----------------------------------------

q[0, :] = 1.0


u = 1.0 # place holder
dt = 0.1 * dx / u


for n in range(0, Nt-1):
  for i in range(1, Np-1):
    q[i, n+1] = q[i, n] - dt / 2.0 / dx * u * (q[i+1, n] - q[i-1, n])


print(q.shape)

xgrid = np.arange(x0, x1, 1)

plt.plot(xgrid, q[:, 0], color = 'k')
plt.plot(xgrid, q[:, 299])

plt.xlim(0, 100)

plt.savefig('fig1.png')

plt.show()



