import numpy as np
import matplotlib.pyplot as plt

def W( x, y, z, h ):
  """
  Gausssian  Smoothing kernel (3D)
  x     is a vector/matrix of x positions
  y     is a vector/matrix of y positions
  z     is a vector/matrix of z positions
  h     is the smoothing length
  w     is the evaluated smoothing function
  """
  
  r = np.sqrt(x**2 + y**2 + z**2)
  
  w = (1.0 / (h*np.sqrt(np.pi)))**3 * np.exp( -r**2 / h**2)
  
  return w
	
	
def gradW( x, y, z, h ):
  """
  Gradient of the Gausssian Smoothing kernel (3D)
  x     is a vector/matrix of x positions
  y     is a vector/matrix of y positions
  z     is a vector/matrix of z positions
  h     is the smoothing length
  wx, wy, wz     is the evaluated gradient
  """
  
  r = np.sqrt(x**2 + y**2 + z**2)
  
  n = -2 * np.exp( -r**2 / h**2) / h**5 / (np.pi)**(3/2)
  wx = n * x
  wy = n * y
  wz = n * z
  
  return wx #, wy, wz # we are in 1D



def getDensity(r, pos, m, h):
  """
  r is sampling location - one single location!!
  pos is SPH particles locations - all particles.
  """
  
  s = 0.0
  
  for i in range(len(pos)):
  
    dx = r[0] - pos[i, 0]
    dy = r[1] - pos[i, 1]
    dz = r[2] - pos[i, 2]
  
    s += m[i] * W( dx, dy, dz, h[i])

  return s # this is the density at the single location r!!! So it is a scalar.


def getPressure(rho, k, n):
  """
  Equation of State
  rho   vector of densities
  k     equat  ion of state constant
  n     polytropic index
  P     pressure
  """
  
  P = k * rho**(1+1/n)
  
  return P




def getAcc(ri, mi, vi, pos, m, h, k, n):

  rhoi = getDensity(ri, pos, m, h)
  Pi = getPressure(rhoi, k, n)

  s = 0.0

  for j in range(len(pos)):
    dx = ri[0] - pos[j, 0]
    dy = ri[1] - pos[j, 1]
    dz = ri[2] - pos[j, 2]

    rhoj = getDensity(pos[j, :], pos, m, h)
    Pj = getPressure(rhoj, k, n)

    s -= mi * (Pi/rhoi/rhoi + Pj/rhoj/rhoj) * gradW(dx, dy, dz, h[j]) - lmbda * ri[0] - nu * vi[0]

  return s


k = 0.1
n = 1.0
nu = 1.0
lmbda = 2.01

p1 = [-0.10, 0.0, 0.0]
p2 = [ 0.10, 0.0, 0.0]

m = [1.0, 1.0]
h = [0.1, 0.1]

pos = np.array([p1, p2])
print(pos.shape)

v = np.zeros_like(pos)
v[0, 0] = 2.0
v[1, 0] = -2.0

rho1 = getDensity(p1, pos, m, h)
print(f'rho1 = {rho1}')


rho2 = getDensity(p2, pos, m, h)
print(f'rho2 = {rho2}')

acc1 = getAcc(pos[0, :], m[0], v[0, :], pos, m, h, k, n)
print(acc1)


acc2 = getAcc(pos[1, :], m[1], v[1, :], pos, m, h, k, n)
print(acc2)

dt = 0.00005

t = 0.0

plt.figure(figsize = (8, 8))

tArr = []
PArr = []

for i in range(2000):

  v[0, 0] += acc1 * dt/2
  v[1, 0] += acc2 * dt/2
  
  pos[0, 0] += v[0, 0] * dt
  pos[1, 0] += v[1, 0] * dt
  
  acc1 = getAcc(pos[0, :], m[0], v[0, :], pos, m, h, k, n)
  acc2 = getAcc(pos[1, :], m[1], v[1, :], pos, m, h, k, n)
  
  v[0, 0] += acc1 * dt/2
  v[1, 0] += acc2 * dt/2
  
  rho1 = getDensity(pos[0, :], pos, m, h)
  P1 = getPressure(rho1, k, n)
  
  rho2 = getDensity(pos[1, :], pos, m, h)
  P2 = getPressure(rho2, k, n)
  
  t += dt
  
  tArr.append(t)
  PArr.append(P1)
  
  #print(pos[0, 0], pos[1, 0])
  #print(acc1, acc2)
  #print(v[0, 0], v[1, 0])
  
  print(P1, P2)
  
  print(t)
  
  if t >= 0.050:
    break
  
  plt.clf()
  
  
  plt.scatter(tArr, PArr, s = 10, color = 'k')
  plt.xlim(0, 0.1)
  plt.ylim(3200, 4200)
  
  
  #plt.scatter(pos[:, 0], pos[:, 1], s = 50, color = 'k')
  #plt.xlim(-1, 1)
  
  plt.pause(0.01)
  plt.draw()

plt.savefig('toy.png')

#plt.show()



