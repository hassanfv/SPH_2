
import numpy as np
import struct
import matplotlib.pyplot as plt



with open("data.bin", "rb") as f:
  N = struct.unpack('i', f.read(4))[0]
  MAX_N = struct.unpack('i', f.read(4))[0]
  MAX_ngb = struct.unpack('i', f.read(4))[0]
  Typ = np.frombuffer(f.read(N * 4), dtype=np.int32)
  x = np.frombuffer(f.read(N * 4), dtype=np.float32)
  y = np.frombuffer(f.read(N * 4), dtype=np.float32)
  z = np.frombuffer(f.read(N * 4), dtype=np.float32)
  h = np.frombuffer(f.read(N * 4), dtype=np.float32)
  ngb = np.frombuffer(f.read(MAX_N * 4), dtype=np.int32)


i = 5

ngb_i = ngb[(i * MAX_ngb):(i*MAX_ngb+MAX_ngb)]

h_i = h[i]

print('h[i] = ', h_i)


rr = np.sqrt((x-x[i])*(x-x[i]) + (y-y[i])*(y-y[i]) + (z-z[i])*(z-z[i]))

sort_rr = np.sort(rr)

#print('sort_rr = ', sort_rr)

nx = np.where(sort_rr <= 2.0*h_i)[0]
print('len(nx) = ', len(nx))

print()
print('sort_rr[len(nx)]/2 = ', sort_rr[len(nx)]/2)
print()

s()

plt.figure(figsize = (8, 8))

plt.scatter(x, y, s = 0.01, color = 'k')
plt.scatter(x[ngb_i], y[ngb_i], s = 10, color = 'red')
plt.scatter([x[i]], [y[i]], s = 20, color = 'lime')


plt.show()



