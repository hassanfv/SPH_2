import numpy as np 

####### Parameters that control the behaviour of chimes-driver 
driver_parameters = {
    "chimes_library_path" : "../chimes/test_runs/libchimes.so", 
    "chimes_data_path" : "../chimes-data", 
    "EqAbundanceTable_filename": None, 
    "IO_mode" : "grid",                 # Options: grid, snapshot 
    "driver_mode" : "noneq_evolution",  # Options: eqm_state, eqm_table, cooling_rates, noneq_evolution 
    "input_file" : None,
    "output_file" : None,
    "hdf5_output_group" : "/", 
    "UV_field" : None,    # Options: None, HM01, HM12, FG20, B87, Colibre, StellarFluxes 
    "shield_mode" : None, # Options: None, read-in, Jeans, Colibre 
    "dust_depletion" : None,  # Options: None, J09, DC16, Colibre 
    "snapshot_type" : None,   # Options: None, GIZMO, AREPO, USER
    "snapshot_cosmo_flag" : 0,      # 0 - Co-moving integration is OFF
                                    # 1 - Co-moving integration is ON
    "snapshot_unitMass_cgs" : 1.989e43,     # 10^10 Msol 
    "snapshot_unitLength_cgs" : 3.0857e21,  # 1 kpc 
    "snapshot_unitVelocity_cgs" : 1.0e5,  # 1 km/s  
    "snapshot_chemistry_array" : None, 
    "snapshot_column_density_array" : "PartType0/ChimesColumnDensity", 
    "snapshot_flux_ion_array" : "PartType0/ChimesFluxIon", 
    "snapshot_flux_G0_array" : "PartType0/ChimesFluxG0", 
    "compute_stellar_fluxes" : 0, 
    "stellar_fluxes_fEsc_ion" : 0.05, 
    "stellar_fluxes_fEsc_G0" : 0.1, 
    "disable_shielding_in_HII_regions" : 0,
    "snapshot_HIIregion_array" : "PartType0/DelayTime_HIIRegion_Cooling", 
    "n_iterations" : 10, 
    "log_T_min" : 2.0, 
    "log_T_max" : 9.0, 
    "delta_log_T" : 0.2, 
    "log_nH_min" : -6.0, 
    "log_nH_max" : 4.0, 
    "delta_log_nH" : 0.2, 
    "log_Z_min" : -3.0, 
    "log_Z_max" : 1.0, 
    "delta_log_Z" : 0.2, 
    "shield_length_factor" : 1.0, 
    "max_shield_length" : 3.086e23,  # 100 kpc 
    "colibre_log_T_min" : 4.0, 
    "colibre_log_T_max" : 5.0, 
    "colibre_scale_MW_ISRF" : 1.0,
    "colibre_ISRF_low_dens_cut_off_redshift" : 7.5, 
    "colibre_use_turbulent_jeans_length" : 1, 
    "colibre_sigma_turb" : 6.0, 
    "colibre_saturate_radiation_field" : 0, 
    "colibre_col_dens_saturate_rad" : 1.3e22, 
    "colibre_saturate_cr_rate" : 0, 
    "colibre_col_dens_saturate_cr" : 3.63e20, 
    "colibre_cr_plaw_index" : 1.0, 
    "radiation_field_normalisation_factor" : 1.0,
    "dust_boost_mode" : 0, # Options: 0 - no dust boost factor 
                           #          1 - constant dust boost factor 
                           #          2 - density-dependent dust boost factor
    "dust_boost_factor" : 1.0, 
    "dust_boost_nH_min_cgs" : 1.0, 
    "dust_boost_nH_max_cgs" : 100.0, 
    "bolometric_AGN_luminosity_cgs" : 1.0e46,
    "distance_to_AGN_kpc" : 1.0,
    "AGN_position_x_kpc" : 0.0, 
    "AGN_position_y_kpc" : 0.0, 
    "AGN_position_z_kpc" : 0.0, 
    "version_date" : "unspecified", 
} 

########################
## Chimes global vars ##
######################## 
# We include only those 
# global vars that can 
# be specified directly 
# in the parameter file. 
global_variable_parameters={
    "redshift" : 0.0, 
    "reionisation_redshift" : 7.5, 
    "use_redshift_dependent_eqm_tables" : 0, 
    "rt_update_flux" : 0, 
    "rt_use_on_the_spot_approx" : 0, 
    "StaticMolCooling" : 1,              
    "Tmol_K" : 1.0e5,    
    "grain_temperature" : 10.0,          
    "cmb_temperature" : 2.725,           
    "relativeTolerance" : 1.0e-4,        
    "absoluteTolerance" : 1.0e-10,       
    "explicitTolerance" : 0.01, 
    "rt_density_absoluteTolerance" : 1.0e-10, 
    "rt_flux_absoluteTolerance" : 1.0e-10, 
    "scale_metal_tolerances" : 1,
    "chimes_debug" : 0, 
    "hybrid_cooling_mode" : 0, 
    "element_included" : (1, 1, 1, 1, 1, 1, 1, 1, 1), 
}

######## Chimes gas vars 
gas_variable_parameters = {
    "cr_rate" : 1.8e-16,
    "TempFloor" : 10.1,
    "divVel" : 1.0,
    "doppler_broad" : 7.1,
    "ForceEqOn" : 0,
    "ThermEvolOn" : 0, 
    "temp_floor_mode" : 0,                
    "InitIonState" : 0, 
    "constant_heating_rate" : 0.0,
    "hydro_timestep" : 3.16e16,  
}

def read_parameters(infile): 
    fd = open(infile, "r")

    element_included = np.ones(9, dtype = np.int64) 

    for line in fd:
        if len(line.strip()) < 1:
            continue
        elif line.strip()[0] == "#": 
            continue 
        else:
            values = line.split()
            if values[0] in driver_parameters:
                # Try to evaluate the parameter as a number,
                # otherwise keep it as a string if that fails. 
                try: 
                    driver_parameters[values[0]] = eval(values[1]) 
                except: 
                    if values[1] == "None": 
                        driver_parameters[values[0]] = None 
                    else: 
                        driver_parameters[values[0]] = values[1]
            elif values[0] in gas_variable_parameters:
                try:
                    gas_variable_parameters[values[0]] = eval(values[1])
                except:
                    gas_variable_parameters[values[0]] = values[1] 
            elif values[0] in global_variable_parameters:
                try:
                    global_variable_parameters[values[0]] = eval(values[1])
                except:
                    global_variable_parameters[values[0]] = values[1]
            elif values[0][:7] == "Include":
                # Flags that switch individual elements on/off.
                # Need to record these in the element_included array,
                # and then we will pass the whole thing over to
                # global_variable_parameters as a tuple 
                if values[0][7:11] == "Carb":
                    element_included[0] = eval(values[1]) 
                elif values[0][7:11] == "Nitr":
                    element_included[1] = eval(values[1]) 
                elif values[0][7:11] == "Oxyg":
                    element_included[2] = eval(values[1]) 
                elif values[0][7:11] == "Neon":
                    element_included[3] = eval(values[1]) 
                elif values[0][7:11] == "Magn":
                    element_included[4] = eval(values[1]) 
                elif values[0][7:11] == "Sili":
                    element_included[5] = eval(values[1]) 
                elif values[0][7:11] == "Sulp":
                    element_included[6] = eval(values[1]) 
                elif values[0][7:11] == "Calc":
                    element_included[7] = eval(values[1]) 
                elif values[0][7:11] == "Iron":
                    element_included[8] = eval(values[1]) 
            else:
                raise KeyError("Parameter %s not recognised. Aborting!" %values[0])

        element_included_tuple = tuple([i for i in element_included])
        global_variable_parameters["element_included"] = element_included_tuple 

    fd.close()

    return driver_parameters, global_variable_parameters, gas_variable_parameters 

def print_parameters(infile, driver_pars, global_pars, gas_pars):
    print("####################")
    print("#### Parameters ####") 
    print("####################")

    print("Parameter file: %s" % (infile, ))
    print(" ") 

    print("Driver parameters:") 
    for key in driver_pars:
        print(key, driver_pars[key])

    print(" ")
    print("Global parameters:")
    for key in global_pars:
        print(key, global_pars[key])

    print(" ")
    print("Gas parameters:")
    for key in gas_pars:
        print(key, gas_pars[key])

    print(" ") 
    
    return
