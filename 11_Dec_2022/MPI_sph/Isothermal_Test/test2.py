

unit_U = 4147431721.875001
UnitTime_in_s = 496890113501.5364
unit_rho = 6.0699462890625e-17 # g/cm3

mH = 1.6726e-24 # gram
kB = 1.3807e-16  # cm2 g s-2 K-1
gamma = 5.0/3.0

#--- Testing a single cooling -----

u = 0.15
u_cgs = u * unit_U

#---- calculating some rough T_init ----
mu_H2 = 2.0 # for a pure H2 gas
T = (gamma - 1.0) * mH / kB * mu_H2 * u_cgs
#---------------------------------------


print('T = ', T)





