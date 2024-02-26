import numpy as np

from phys_const import proton_mass_cgs, boltzmann_cgs, newton_G_cgs 

# Compute Jeans length assuming 
# hydrogen mass fraction XH = 0.7 
# mean molecular weight mu = 1 
def compute_jeans_shield_length(T, nH, shield_length_factor, max_shield_length):
    LJ = 5. * 3.086e18 * 0.5
    #LJ = shield_length_factor * np.sqrt(((5.0 / 3.0) * np.pi * boltzmann_cgs * 0.7) / (newton_G_cgs * (proton_mass_cgs ** 2.0))) * np.sqrt(T / nH)
    #print(f'{T:.1f}, {nH:.3f}, {(LJ/3.086e18):.2f} pc')
    return min(LJ, max_shield_length)

# Compute reference column density 
# used in the COLIBRE model 
# (Ploeckinger et al. 2020). 
def compute_colibre_Nref(T, nH, XH, max_shield_length, colibre_log_T_min, colibre_log_T_max, use_turbulence, sigma_turb): 
    l_max = max_shield_length 
    nH_min = 1.0e-8 
    N_max = 1.0e24 
    N_min = l_max * nH_min 
    
    # Assume mean molecular weight == 1 
    N_J_thermal = nH * np.sqrt((5.0 / 3.0) * boltzmann_cgs * T / (newton_G_cgs * (nH * proton_mass_cgs / XH) * proton_mass_cgs)) 

    if use_turbulence == 1: 
        N_J_turb = np.sqrt((5.0 / 3.0) * XH * nH / (proton_mass_cgs * newton_G_cgs)) * (sigma_turb * 1.0e5)
        N_J = max(N_J_thermal, N_J_turb) 
    else: 
        N_J = N_J_thermal 

    N_ref_prime = min([N_J, l_max * nH, N_max]) 
    N_ref = 10.0 ** (np.log10(N_ref_prime) - ((np.log10(N_ref_prime) - np.log10(N_min)) / (1.0 + np.exp(-5.0 * (np.log10(T) - ((colibre_log_T_min + colibre_log_T_max) / 2.0)))))) 

    return N_ref 

# Compute the shielding length 
# as used in the COLIBRE model 
# (Ploeckinger et al. in prep). 
def compute_colibre_shield_length(T, nH, XH, shield_length_factor, max_shield_length, colibre_log_T_min, colibre_log_T_max, use_turbulence, sigma_turb): 
    N_ref = compute_colibre_Nref(T, nH, XH, max_shield_length, colibre_log_T_min, colibre_log_T_max, use_turbulence, sigma_turb)

    return shield_length_factor * N_ref / nH 
