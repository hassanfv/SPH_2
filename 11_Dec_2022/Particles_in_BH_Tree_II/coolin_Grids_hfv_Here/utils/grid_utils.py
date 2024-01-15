import numpy as np 
import os 
import h5py
import sys

from chimes_utils import set_initial_chemistry_abundances 
from chimes_classes import *
from shielding_utils import compute_jeans_shield_length, compute_colibre_shield_length 

# Create a series of 'gas particles' spanning 
# a 3-D grid of density, temperature and metallicity 
def create_grid(driver_pars, global_pars, init_ion_state): 
    T = 10.0 ** np.arange(driver_pars["log_T_min"], driver_pars["log_T_max"] + (driver_pars["delta_log_T"] / 10.0), driver_pars["delta_log_T"]) 
    nH = 10.0 ** np.arange(driver_pars["log_nH_min"], driver_pars["log_nH_max"] + (driver_pars["delta_log_nH"] / 10.0), driver_pars["delta_log_nH"]) 
    Z = 10.0 ** np.arange(driver_pars["log_Z_min"], driver_pars["log_Z_max"] + (driver_pars["delta_log_Z"] / 10.0), driver_pars["delta_log_Z"]) 

    dim_T = len(T) 
    dim_nH = len(nH) 
    dim_Z = len(Z) 

    N_tot = dim_T * dim_nH * dim_Z 
    temperature_arr = np.zeros(N_tot) 
    nH_arr = np.zeros(N_tot) 
    metallicity_arr = np.zeros((N_tot, 11)) 
    shieldLength_arr = np.zeros(N_tot) 

    for i in range(dim_T): 
        for j in range(dim_nH): 
            for k in range(dim_Z): 
                array_index = (i * dim_nH * dim_Z) + (j * dim_Z) + k 
                temperature_arr[array_index] = T[i] 
                nH_arr[array_index] = nH[j] 
                metallicity_arr[array_index, :] = compute_metallicity_array(Z[k]) 
                
                if driver_pars["shield_mode"] == None: 
                    shieldLength_arr[array_index] = 1.0 
                elif driver_pars["shield_mode"] == "Jeans": 
                    # Compute Jeans length assuming 
                    # hydrogen mass fraction XH = 0.7 
                    # mean molecular weight mu = 1 
                    shieldLength_arr[array_index] = compute_jeans_shield_length(T[i], nH[j], driver_pars["shield_length_factor"], driver_pars["max_shield_length"]) 
                elif driver_pars["shield_mode"] == "Colibre": 
                    XH = 1.0 - (metallicity_arr[array_index, 0] + metallicity_arr[array_index, 1]) 
                    shieldLength_arr[array_index] = compute_colibre_shield_length(T[i], nH[j], XH, driver_pars["shield_length_factor"], driver_pars["max_shield_length"], driver_pars["colibre_log_T_min"], driver_pars["colibre_log_T_max"], driver_pars["colibre_use_turbulent_jeans_length"], driver_pars["colibre_sigma_turb"])
                else: 
                    raise Exception("shield_mode %s not recognised. Aborting" % (driver_pars["shield_mode"], )) 

    init_chem_arr = set_initial_chemistry_abundances(metallicity_arr, global_pars, init_ion_state) 
    
    print("Cooling table grid:") 
    print("%.2f <= log10(T) <= %.2f" % (min(np.log10(temperature_arr)), max(np.log10(temperature_arr)))) 
    print("%.2f <= log10(nH) <= %.2f" % (min(np.log10(nH_arr)), max(np.log10(nH_arr)))) 
    print("%.2f <= log10(Z/Zsol) <= %.2f\n" % (min(np.log10(metallicity_arr[:, 0] / 0.0129)), max(np.log10(metallicity_arr[:, 0] / 0.0129))))
    sys.stdout.flush() 
    
    return nH_arr, temperature_arr, metallicity_arr, shieldLength_arr, init_chem_arr 

# For a given metallicity relative to solar, 
# compute the mass fractions of individual elements 
# Note: we define solar abundances as given in 
# Table 1 of Wiersma et al., 2009, MNRAS, 393, 99 
def compute_metallicity_array(Z_sol): 
    solar_abundances = np.zeros(11) 

    # Solar mass fractions 
    solar_abundances[0] = 0.0129  # Z 
    solar_abundances[1]=0.2806    # He 
    solar_abundances[2]=2.07e-3   # C 
    solar_abundances[3]=8.36e-4   # N 
    solar_abundances[4]=5.49e-3   # O 
    solar_abundances[5]=1.41e-3   # Ne 
    solar_abundances[6]=5.91e-4   # Mg 
    solar_abundances[7]=6.83e-4   # Si 
    solar_abundances[8]=4.09e-4   # S 
    solar_abundances[9]=6.44e-5   # Ca 
    solar_abundances[10]=1.1e-3   # Fe 

    output_array = solar_abundances * Z_sol 
    
    # Account for a primordial He abundance 
    output_array[1] = 0.25 + ((solar_abundances[1] - 0.25) * Z_sol) 

    return output_array 

# Routine to check whether output 
# arrays already exist in the 
# output file, for IO_mode == grid. 
def grid_check_output_arrays(driver_pars): 
    return_value = False 

    if os.path.exists(driver_pars["output_file"]): 
        with h5py.File(driver_pars["output_file"], 'r') as h5file: 
            if driver_pars["driver_mode"] == "eqm_state" or driver_pars["driver_mode"] == "eqm_table": 
                array_name_list = ["Abundances", "TableBins/Temperatures", "TableBins/Densities", "TableBins/Metallicities", "TableBins/N_Temperatures", "TableBins/N_Densities", "TableBins/N_Metallicities", "TableBins/N_species"] 
            elif driver_pars["driver_mode"] == "cooling_rates": 
                array_name_list = ["log_cooling_rate", "log_heating_rate", "TableBins/Temperatures", "TableBins/Densities", "TableBins/Metallicities", "TableBins/N_Temperatures", "TableBins/N_Densities", "TableBins/N_Metallicities"] 
            elif driver_pars["driver_mode"] == "noneq_evolution": 
                array_name_list = ["AbundanceEvolution", "TemperatureEvolution", "TimeArray_seconds", "TableBins/Temperatures", "TableBins/Densities", "TableBins/Metallicities", "TableBins/N_Temperatures", "TableBins/N_Densities", "TableBins/N_Metallicities"] 
            else: 
                raise Exception("ERROR: IO_mode %s cannot be run with driver_mode %s. Aborting." % (driver_pars["IO_mode"], driver_pars["driver_mode"])) 
                
            for array_name in array_name_list: 
                node_test = array_name in h5file 
                
                if node_test == True: 
                    return_value = True
                    print("%s already present in file %s" % (array_name, driver_pars["output_file"]))
                    sys.stdout.flush()

    return return_value 
