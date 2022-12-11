

unit_U = 9537129466.43166
UnitTime_in_s = 3.16e13 # == 1Myr
unit_rho = 1.5e-20 # g/cm3

X = 0.75
Y = 0.25
mH = 1.6726e-24 # gram
kB = 1.3807e-16  # cm2 g s-2 K-1
gamma = 5./3.

#--- Testing a single cooling -----

rho = 0.01
u = 0.422
dt = 0.005

rho_cgs = rho * unit_rho
u_cgs = u * unit_U
dt_cgs = dt * UnitTime_in_s

#---- calculating some rough T_init ----
y = 0.5 # Just approximation to have some estimate of T_init (doesn't need to be exact).
mu_approx = X/2. + X*y/2. + Y/4.
T_init = (gamma - 1.0) * mH / kB * mu_approx * u_cgs
#---------------------------------------

print(T_init)
