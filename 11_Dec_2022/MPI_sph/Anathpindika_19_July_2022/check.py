
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time


unitTime_in_Myr =  3.5174499013053913 # Myr


filz = np.sort(glob.glob('./Outputs/*.pkl'))
#filz = np.sort(glob.glob('./Outputs_130k_M_2000_b_0.28/*.pkl'))

#j = -1

plt.ion()
fig, ax = plt.subplots(figsize = (10, 6))

kb = ''

for j in range(0, len(filz), 10):

    print('j = ', j)

    with open(filz[j], 'rb') as f:
        data = pickle.load(f)


    r = data['pos']
    h = data['h']

    print('h = ', np.sort(h))

    #print(r.shape)

    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]
    t = data['current_t']
    rho = data['rho']

    if j == 0: # only for j=0 because for other j that particle will move obviously and is not at the center anymore.
        rr = (x*x + y*y + z*z)**0.5
        x0 = 0.0
        tmp = np.abs(rr - x0)
        ncen = np.where(tmp == np.min(tmp))[0] # index of the particle closest to the center of the left-side cloud.
    
	
    ax.cla()

    ax.scatter(x, y, s = 0.01, color = 'black')
    xyrange = 1.2

    ax.axis(xmin = -1.2, xmax = 3.2)
    ax.axis(ymin = -1.2, ymax = 1.5)


    ax.axvline(x =  0.0, linestyle = '--', color = 'blue')
    #ax.axvline(x =  2.0, linestyle = '--', color = 'blue')
    ax.axvline(x =  x[ncen], linestyle = '--', color = 'red')

    R_0 = 4.81 # Take this from the terminal after running the step_2_IC....py !
    ax.set_title(f't = {np.round(t*unitTime_in_Myr,4)}        d = {round(x[ncen][0]*R_0, 2)} pc')
    fig.canvas.flush_events()
    time.sleep(0.01)


    kb =readchar.readkey()

    if kb == 'q':
        break

plt.savefig('1111.png')







