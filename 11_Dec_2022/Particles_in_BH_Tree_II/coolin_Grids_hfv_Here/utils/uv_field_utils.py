import numpy as np
import sys

# Stellar luminosity per stellar mass in 
# the 6-13.6 eV band as a function of 
# stellar age, in units of Habing cm^2 Msol^-1. 
# Based on fits to Starburst 99 models; see 
# Richings et al (in prep) for details. 
def stellar_luminosity_G0(stellar_age_Myr): 
    output_array = np.zeros(len(stellar_age_Myr)) 

    incl_first = (stellar_age_Myr < 4.1) 
    output_array[incl_first] = np.exp(108.1 + (0.17 * (np.float64(stellar_age_Myr[incl_first]) ** 0.92))) 

    incl_second = ~incl_first 
    output_array[incl_second] = 6.4e37 * ((1.77e6 / np.float64(stellar_age_Myr[incl_second])) ** 1.67) * ((1.0 + ((np.float64(stellar_age_Myr[incl_second]) / 1.77e6) ** 28.2)) ** 1.65) 

    # The fit parameters in the paper give the 
    # luminosity in photons s^-1 Msol^-1. We 
    # need to convert this to 
    # Habing cm^2 Msol^-1. The average photon 
    # energy in the 6-13.6 eV band is 
    # <h nu> = 10.0 eV 
    # for the Starburst99 models.
    output_array /= 9.9053e7 

    return output_array 

# Stellar luminosity per stellar mass in 
# the >13.6 eV band as a function of stellar 
# age, in units of photons s^-1 Msol^-1. 
# Based on fits to Starburst 99 models; 
# see Richings et al (in prep) for details. 
def stellar_luminosity_ion(stellar_age_Myr): 
    output_array = np.zeros(len(stellar_age_Myr)) 

    incl_first = (stellar_age_Myr < 3.7) 
    output_array[incl_first] = np.exp(107.2 + (0.11 * (np.float64(stellar_age_Myr[incl_first]) ** 0.97))) 

    incl_second = ~incl_first 
    output_array[incl_second] = 3.3e21 * ((6.89e5 / np.float64(stellar_age_Myr[incl_second])) ** 4.79) * ((1.0 + ((np.float64(stellar_age_Myr[incl_second]) / 6.89e5) ** 1.12)) ** (-1.7e4))  

    return output_array 

# Upper bounds of the stellar age bins 
# used for the stellar fluxes in Richings 
# et al (in prep). Note that there is also 
# an 8th bin for stars > 100 Myr. 
log_stellar_age_Myr_bin_upper = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0]) 

# Calculates the stellar fluxes in each 
# stellar age bin for all gas particles. 
# See Richings et al (in prep) for details. 
def compute_stellar_fluxes(gas_coords_cgs, star_coords_cgs, star_mass_Msol, star_age_Myr, fEsc_ion, fEsc_G0, rank): 
    Npart_gas = len(gas_coords_cgs) 
    Npart_star = len(star_coords_cgs) 

    ChimesFluxIon = np.zeros((Npart_gas, len(log_stellar_age_Myr_bin_upper) + 1), dtype = np.float32) 
    ChimesFluxG0 = np.zeros((Npart_gas, len(log_stellar_age_Myr_bin_upper) + 1), dtype = np.float32) 

    log_star_age_Myr = np.log10(star_age_Myr) 

    for idx_gas in range(Npart_gas):
        if rank == 0:
            if idx_gas % 100 == 0:
                print("Rank %d: Computing stellar flux for particle %d of %d" % (rank, idx_gas, Npart_gas))
                sys.stdout.flush() 
        
        delta_coords = np.float64((star_coords_cgs - gas_coords_cgs[idx_gas, :])) 
        star_distance_squared_cgs = (delta_coords[:, 0] ** 2.0) + (delta_coords[:, 1] ** 2.0) + (delta_coords[:, 2] ** 2.0) 
        
        for idx_age in range(len(log_stellar_age_Myr_bin_upper) + 1): 
            if idx_age == 0: 
                star_incl = (log_star_age_Myr <= log_stellar_age_Myr_bin_upper[idx_age]) 
            elif idx_age == len(log_stellar_age_Myr_bin_upper): 
                star_incl = (log_star_age_Myr > log_stellar_age_Myr_bin_upper[idx_age - 1]) 
            else: 
                star_incl = ((log_star_age_Myr <= log_stellar_age_Myr_bin_upper[idx_age]) & (log_star_age_Myr > log_stellar_age_Myr_bin_upper[idx_age - 1])) 

            star_L_ion = fEsc_ion * stellar_luminosity_ion(star_age_Myr[star_incl]) * star_mass_Msol[star_incl] 
            star_L_G0 = fEsc_G0 * stellar_luminosity_G0(star_age_Myr[star_incl]) * star_mass_Msol[star_incl] 

            one_over_r2 = 1.0 / star_distance_squared_cgs[star_incl]

            ChimesFluxIon[idx_gas, idx_age] = np.sum(star_L_ion * one_over_r2) / (4.0 * np.pi)
            ChimesFluxG0[idx_gas, idx_age] = np.sum(star_L_G0 * one_over_r2) / (4.0 * np.pi)

    return np.float32(ChimesFluxIon), np.float32(ChimesFluxG0) 

# Calculate ISRF and cr_rate 
# for the COLIBRE model 
def compute_colibre_ISRF(myGasVars, myGlobalVars, spectra_table, N_ref_over_NH0_rad, N_ref_over_NH0_cr, colibre_scale_MW_ISRF, cr_rate, ISRF_low_dens_cut_off_redshift, cr_plaw_index): 
    J_over_J0 = colibre_scale_MW_ISRF * (N_ref_over_NH0_rad ** 1.4) 
    myGasVars.isotropic_photon_density[1] = spectra_table.isotropic_photon_density[1] 

    if (myGlobalVars.redshift > ISRF_low_dens_cut_off_redshift) and (J_over_J0 > 0.0): 
        myGasVars.isotropic_photon_density[1] *= 10.0 ** (-20.0 - ((-20.0 - np.log10(J_over_J0)) / (1.0 + np.exp(-2.0 * (np.log10(myGasVars.nH_tot) + 4.0)))))
    else: 
        myGasVars.isotropic_photon_density[1] *= J_over_J0 

    myGasVars.cr_rate = cr_rate * colibre_scale_MW_ISRF * (N_ref_over_NH0_cr ** cr_plaw_index) 
    
    return

# Calculate the isotropic photon density
# for the average quasar spectrum from
# Sazonov et al (2004), given the 
# bolometric AGN luminosity and distance
# from the AGN. 
def compute_AGN_isotropic_photon_density(L_AGN_cgs, r_kpc):
    return 5.39903 * (L_AGN_cgs / 1.0e46) / (r_kpc ** 2.0) 
