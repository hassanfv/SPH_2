import numpy as np 

# Arrays of the fit coefficients, given in 
# Table 4 of Jenkins (2009). 
# In the order: Ax, Bx, zx. 
J09_pars_C = np.array([-0.101, -0.193, 0.803])
J09_pars_N = np.array([0.0, -0.109, 0.55]) 
J09_pars_O = np.array([-0.225, -0.145, 0.598]) 
J09_pars_Mg = np.array([-0.997, -0.800, 0.531]) 
J09_pars_Si = np.array([-1.136, -0.570, 0.305])
J09_pars_P = np.array([-0.945, -0.166, 0.488]) 
J09_pars_Cl = np.array([-1.242, -0.314, 0.609]) 
J09_pars_Ti = np.array([-2.048, -1.957, 0.43]) 
J09_pars_Cr = np.array([-1.447, -1.508, 0.47]) 
J09_pars_Mn = np.array([-0.857, -1.354, 0.52]) 
J09_pars_Fe = np.array([-1.285, -1.513, 0.437]) 
J09_pars_Ni = np.array([-1.490, -1.829, 0.599]) 
J09_pars_Cu = np.array([-0.710, -1.102, 0.711]) 
J09_pars_Zn = np.array([-0.610, -0.279, 0.555]) 
J09_pars_Ge = np.array([-0.615, -0.725, 0.69]) 
J09_pars_Kr = np.array([-0.166, -0.332, 0.684])

# Arrays of the fit coefficents given in 
# Table 3 of De Cia et al (2016). 
# In the order: A2, B2. 
DC16_pars_O = np.array([-0.02, -0.15])
DC16_pars_Mg = np.array([-0.03, -0.61])
DC16_pars_Si = np.array([-0.03, -0.63])
DC16_pars_P = np.array([0.01, -0.10])
DC16_pars_S = np.array([-0.04, -0.28])
DC16_pars_Cr = np.array([0.15, -1.32])
DC16_pars_Mn = np.array([0.04, -0.95])
DC16_pars_Fe = np.array([-0.01, -1.26]) 
DC16_pars_Zn = np.array([0.0, -0.27])

# Atomic masses (in the same order as above)
atomic_mass = np.array([12.0, 14.0, 16.0, 24.0, 28.0, 31.0, 35.0, 48.0, 52.0, 55.0, 56.0, 59.0, 64.0, 65.0, 73.0, 84.0])
atomic_mass_S = 32.0 

# Cloudy default abundances
solar_abundance = 12.0 + np.log10(np.array([2.45e-4, 8.51e-5, 4.90e-4, 3.47e-5, 3.47e-5, 3.20e-7, 1.91e-7, 1.05e-7, 4.68e-7, 2.88e-7, 2.82e-5, 1.78e-6, 1.62e-8, 3.98e-8, 5.01e-9, 2.29e-9]))
solar_abundance_S = 1.86e-5 

# Hydrogen mass fraction at solar metallicity 
solar_XH = 0.7065 

def J09DC16_compute_Fstar(nH):
    # Returns the parameter F_star, as a function
    # nH (in cgs units). Uses the best-fit relation
    # from Fig. 16 of Jeankins (2009). 
    Fstar = 0.772 + (np.log10(nH) * 0.461)
    if Fstar > 1.0:
        return 1.0
    else:
        return Fstar

def J09_element_linear_fit(Fstar, pars, extrapolate = 1):
    # Returns [X_gas / H]fit, i.e. log10 of 
    # the fraction of the element that 
    # remains in the gas phase, as given by 
    # equation 10 in Jenkins (2009). 
    # pars contains the fit coefficients. 
    Ax = pars[0]
    Bx = pars[1]
    zx = pars[2]

    if extrapolate == 0: 
        # Set all metals to be in the gas phase for Fstar < 0 
        if Fstar < 0.0:
            return 0.0
        else: 
            return Bx + (Ax * (Fstar - zx))
    else:
        # Smoothly extrapolate depltion factors at Fstar < 0
        # until they go to zero
        output = Bx + (Ax * (Fstar - zx))
        if output > 0.0:
            return 0.0
        else:
            return output 

def DC16_element_linear_fit(Fstar, pars):
    # Returns [X_gas / H]fit, as given by 
    # equation 5 in De Cia et al (2016). 
    A2 = pars[0]
    B2 = pars[1]

    # Note: DC16 always allows Fstar < 0
    Zn_over_Fe = (Fstar + 1.5) / 1.48 
    output = A2 + (B2 * Zn_over_Fe) 
    
    if output > 0.0:
        return 0.0
    else:
        return output 
        
def J09DC16_compute_dust_to_gas_ratio(nH, model):
    # computes the ratio of dust mass (summing over all
    # of the above elements) to gas mass, for a given nH.

    Fstar = J09DC16_compute_Fstar(nH)
    dust_to_gas = 0.0 

    # Sum dust to gas mass ratios over all elements
    if model == "J09":
        # Use only Jenkins (2009) 
        dust_to_gas += atomic_mass[2] * solar_XH * (10.0 ** (solar_abundance[2] - 12.0)) * (1.0 - (10.0 ** J09_element_linear_fit(Fstar, J09_pars_O)))
        dust_to_gas += atomic_mass[3] * solar_XH * (10.0 ** (solar_abundance[3] - 12.0)) * (1.0 - (10.0 ** J09_element_linear_fit(Fstar, J09_pars_Mg))) 
        dust_to_gas += atomic_mass[4] * solar_XH * (10.0 ** (solar_abundance[4] - 12.0)) * (1.0 - (10.0 ** J09_element_linear_fit(Fstar, J09_pars_Si))) 
        dust_to_gas += atomic_mass[5] * solar_XH * (10.0 ** (solar_abundance[5] - 12.0)) * (1.0 - (10.0 ** J09_element_linear_fit(Fstar, J09_pars_P))) 
        dust_to_gas += atomic_mass[8] * solar_XH * (10.0 ** (solar_abundance[8] - 12.0)) * (1.0 - (10.0 ** J09_element_linear_fit(Fstar, J09_pars_Cr))) 
        dust_to_gas += atomic_mass[9] * solar_XH * (10.0 ** (solar_abundance[9] - 12.0)) * (1.0 - (10.0 ** J09_element_linear_fit(Fstar, J09_pars_Mn))) 
        dust_to_gas += atomic_mass[10] * solar_XH * (10.0 ** (solar_abundance[10] - 12.0)) * (1.0 - (10.0 ** J09_element_linear_fit(Fstar, J09_pars_Fe))) 
        dust_to_gas += atomic_mass[13] * solar_XH * (10.0 ** (solar_abundance[13] - 12.0)) * (1.0 - (10.0 ** J09_element_linear_fit(Fstar, J09_pars_Zn)))
    elif model == "DC16":
        # Use De Cia et al (2016) where available, 
        # otherwise use Jenkins (2009). 
        dust_to_gas += atomic_mass[2] * solar_XH * (10.0 ** (solar_abundance[2] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_O)))
        dust_to_gas += atomic_mass[3] * solar_XH * (10.0 ** (solar_abundance[3] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Mg))) 
        dust_to_gas += atomic_mass[4] * solar_XH * (10.0 ** (solar_abundance[4] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Si))) 
        dust_to_gas += atomic_mass[5] * solar_XH * (10.0 ** (solar_abundance[5] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_P))) 
        dust_to_gas += atomic_mass[8] * solar_XH * (10.0 ** (solar_abundance[8] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Cr))) 
        dust_to_gas += atomic_mass[9] * solar_XH * (10.0 ** (solar_abundance[9] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Mn))) 
        dust_to_gas += atomic_mass[10] * solar_XH * (10.0 ** (solar_abundance[10] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Fe))) 
        dust_to_gas += atomic_mass[13] * solar_XH * (10.0 ** (solar_abundance[13] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Zn)))
        
        dust_to_gas += atomic_mass_S * solar_XH * (10.0 ** (solar_abundance_S - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_S)))
    else: 
        raise Exception("ERROR: Dust Model %s not recognised. Aborting." % (model, )) 
        
    # The following are only in Jenkins (2009) 
    dust_to_gas += atomic_mass[0] * solar_XH * (10.0 ** (solar_abundance[0] - 12.0)) * (1.0 - (10.0 ** J09_element_linear_fit(Fstar, J09_pars_C))) 
    dust_to_gas += atomic_mass[1] * solar_XH * (10.0 ** (solar_abundance[1] - 12.0)) * (1.0 - (10.0 ** J09_element_linear_fit(Fstar, J09_pars_N))) 
    dust_to_gas += atomic_mass[6] * solar_XH * (10.0 ** (solar_abundance[6] - 12.0)) * (1.0 - (10.0 ** J09_element_linear_fit(Fstar, J09_pars_Cl))) 
    dust_to_gas += atomic_mass[7] * solar_XH * (10.0 ** (solar_abundance[7] - 12.0)) * (1.0 - (10.0 ** J09_element_linear_fit(Fstar, J09_pars_Ti))) 
    dust_to_gas += atomic_mass[11] * solar_XH * (10.0 ** (solar_abundance[11] - 12.0)) * (1.0 - (10.0 ** J09_element_linear_fit(Fstar, J09_pars_Ni))) 
    dust_to_gas += atomic_mass[12] * solar_XH * (10.0 ** (solar_abundance[12] - 12.0)) * (1.0 - (10.0 ** J09_element_linear_fit(Fstar, J09_pars_Cu))) 
    dust_to_gas += atomic_mass[14] * solar_XH * (10.0 ** (solar_abundance[14] - 12.0)) * (1.0 - (10.0 ** J09_element_linear_fit(Fstar, J09_pars_Ge))) 
    dust_to_gas += atomic_mass[15] * solar_XH * (10.0 ** (solar_abundance[15] - 12.0)) * (1.0 - (10.0 ** J09_element_linear_fit(Fstar, J09_pars_Kr))) 
    
    return dust_to_gas

# Deplete metals based on the Jenkins (2009)
# and De Cia et al (2016) depletion factors.
# See Richings et al. (in prep) for details of
# how this is implemented as a function of
# gas density.
def J09DC16_set_depletion_factors(myGasVars, model): 
    myGasVars.dust_ratio = myGasVars.metallicity 

    if myGasVars.temperature < 1.0e6:         
        dust_to_gas_saturated = J09DC16_compute_dust_to_gas_ratio(1000.0, model) 
        myGasVars.dust_ratio *= J09DC16_compute_dust_to_gas_ratio(myGasVars.nH_tot, model) / dust_to_gas_saturated 

        Fstar = J09DC16_compute_Fstar(myGasVars.nH_tot)
        
        if model == "J09": 
            # Use only Jenkins (2009) 
            myGasVars.element_abundances[3] *= 10.0 ** J09_element_linear_fit(Fstar, J09_pars_O)
            myGasVars.element_abundances[5] *= 10.0 ** J09_element_linear_fit(Fstar, J09_pars_Mg)
            myGasVars.element_abundances[6] *= 10.0 ** J09_element_linear_fit(Fstar, J09_pars_Si)
            myGasVars.element_abundances[9] *= 10.0 ** J09_element_linear_fit(Fstar, J09_pars_Fe)

        elif model == "DC16": 
            # Use De Cia et al (2016) where available, 
            # otherwise use Jenkins (2009). 
            myGasVars.element_abundances[3] *= 10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_O)
            myGasVars.element_abundances[5] *= 10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Mg)
            myGasVars.element_abundances[6] *= 10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Si)
            myGasVars.element_abundances[7] *= 10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_S)
            myGasVars.element_abundances[9] *= 10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Fe)

        else: 
            raise Exception("ERROR: Dust Model %s not recognised. Aborting." % (model, )) 

        # The following are only in Jenkins (2009) 
        myGasVars.element_abundances[1] *= 10.0 ** J09_element_linear_fit(Fstar, J09_pars_C)
        myGasVars.element_abundances[2] *= 10.0 ** J09_element_linear_fit(Fstar, J09_pars_N)

        return 

# Deplete metals based on the COLIBRE 
# model (see Ploeckinger et al in prep)
def colibre_set_depletion_factors(myGasVars, N_ref_over_N_H0): 
    factor =  min([N_ref_over_N_H0 ** 1.4, 1.0]) 

    myGasVars.dust_ratio = myGasVars.metallicity 
    myGasVars.dust_ratio *= factor 

    f_dust0_C = 0.34385
    f_dust0_O = 0.31766
    f_dust0_Mg = 0.94338
    f_dust0_Si = 0.94492
    f_dust0_Ca = 0.9999
    f_dust0_Fe = 0.99363
    
    myGasVars.element_abundances[1] *= (1.0 - (f_dust0_C * factor)) 
    myGasVars.element_abundances[3] *= (1.0 - (f_dust0_O * factor)) 
    myGasVars.element_abundances[5] *= (1.0 - (f_dust0_Mg * factor)) 
    myGasVars.element_abundances[6] *= (1.0 - (f_dust0_Si * factor)) 
    myGasVars.element_abundances[8] *= (1.0 - (f_dust0_Ca * factor)) 
    myGasVars.element_abundances[9] *= (1.0 - (f_dust0_Fe * factor)) 

    return 
