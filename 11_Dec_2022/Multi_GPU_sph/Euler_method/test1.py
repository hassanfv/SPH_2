
import numpy as np
import matplotlib.pyplot as plt


def func(t, y):

    return y


def euler_forward(t_i, y_i, h):

    return y_i + h * func(t_i, y_i)


def Adam_Bashforth(t_i_1, y_i_1, t_i, y_i, h):

    return y_i_1 + 3./2. * h * func(t_i_1, y_i_1) - 0.5 * h * func(t_i, y_i)



y_i = 1.0 # initial value at t_i = 0
t_i = 0

h = 0.1 # time step

res = []

for i in range(100):

    res.append([t_i, y_i])

    y_next_i = y_i + h * func(t_i, y_i)
    
    y_i = y_next_i

    t_i += h

res = np.array(res)

t = res[:, 0]
y = res[:, 1]

plt.plot(t, y, color = 'k')
plt.plot(t, np.exp(t), color = 'red')




y_i = 1.0 # initial value at t_i = 0
t_i = 0

h = 0.1 # time step

t_i_1 = t_i + h

y_i_1 = y_i + h * func(t_i, y_i)

res = []

for i in range(100):

    res.append([t_i_1, y_i_1])
    
    y_next_i = y_i_1 + 3./2. * h * func(t_i_1, y_i_1) - 0.5 * h * func(t_i, y_i)
    
    y_i = y_i_1
    y_i_1 = y_next_i
    
    t_i = t_i_1
    t_i_1 += h

res = np.array(res)
t = res[:, 0]
y = res[:, 1] 

plt.plot(t, y, color = 'blue')

plt.show()





