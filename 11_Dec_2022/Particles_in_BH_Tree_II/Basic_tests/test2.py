
import numpy as np
import matplotlib.pyplot as plt


def plot_h(dt):
  final_t = 10.0
  #dt = 0.2

  N = np.int32(final_t/dt)

  q = np.zeros(N)
  t = np.zeros(N)

  t[0] = 0.0
  q[0] = 1.0

  for n in range(0, N-1):

    q[n+1] = q[n] - q[n]*q[n] * dt
    t[n+1] = t[n] + dt


  plt.plot(t, q)
  
plot_h(0.1)
plot_h(0.2)
plot_h(0.3)
plot_h(0.4)
plot_h(0.5)


plt.xlim(0.0, 10.0)
plt.ylim(-0.1, 1.1)

plt.show()



