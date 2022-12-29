
import numpy as np
from photolibs2 import *
import matplotlib.pyplot as plt

XH = 0.76
mH = 1.6726e-24 # gram

UnitTime_in_yrs = 300 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
dt_t  = UnitTime_in_yrs * 3600. * 24. * 365.24

nHcgs = 0.1 # cm^-3   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
rhot = nHcgs * mH / XH


if False:

	T = 100000. # K

	uad = convert_Temp_to_u(T, nHcgs, XH)

	print('nHcgs = ', nHcgs)
	print(f'uad = {uad:.3E}')



print('UnitTime_in_yrs = ', UnitTime_in_yrs)




if True:

	uad = 2.4789E+13;
	rhot = 1.6726e-22/0.76;  # 2.4968E-22
	#delta = 2.2955E+13;
	#uxx = uad - delta;
	
	nHcgs = XH/mH * rhot
	print()
	print('nHcgs = ', nHcgs)

	TempX = convert_u_to_temp_h(uad, nHcgs, XH)
	print()
	print(f'Corresp. Temp = {TempX}')

	ux = DoCooling_h(rhot, uad, dt_t, XH)

	delta_u = uad - ux
	print()
	print(f'delta_u = {delta_u:.3E}')

	print()
	print(f'u (before) = {uad:.3E}')
	print(f'u (After) = {ux:.3E}')







