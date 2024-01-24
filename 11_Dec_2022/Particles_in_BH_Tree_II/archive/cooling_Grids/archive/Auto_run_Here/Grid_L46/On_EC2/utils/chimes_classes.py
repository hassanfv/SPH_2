import ctypes as ct
import sys

# The following defines whether CHIMES is in single 
# or double precision. This needs to be consistent 
# with the ChimesFloat type defined in the CHIMES 
# library
use_single = False 
if len(sys.argv) > 2:
    for arg_str in sys.argv:
        if arg_str == "--single-precision":
            use_single = True
            break 

if use_single == True:
    ChimesFloat = ct.c_float
else: 
    ChimesFloat = ct.c_double 

# The following structures will be needed by the CHIMES library 

# Gas variables structure
class gasVariables(ct.Structure):
    _fields_ = [("element_abundances", ChimesFloat * 10), 
                ("nH_tot", ChimesFloat), 
                ("temperature", ChimesFloat), 
                ("TempFloor", ChimesFloat), 
                ("divVel", ChimesFloat), 
                ("doppler_broad", ChimesFloat), 
                ("isotropic_photon_density", ct.POINTER(ChimesFloat)), 
                ("flux_reduction_factor", ct.POINTER(ChimesFloat)), 
                ("G0_parameter", ct.POINTER(ChimesFloat)), 
                ("H2_dissocJ", ct.POINTER(ChimesFloat)), 
                ("rt_source_term", ct.POINTER(ChimesFloat)), 
                ("cr_rate", ChimesFloat), 
                ("metallicity", ChimesFloat), 
                ("dust_ratio", ChimesFloat), 
                ("dust_boost_factor", ChimesFloat), 
                ("cell_size", ChimesFloat), 
                ("hydro_timestep", ChimesFloat), 
                ("ForceEqOn", ct.c_int), 
                ("ThermEvolOn", ct.c_int), 
                ("temp_floor_mode", ct.c_int), 
                ("InitIonState", ct.c_int), 
                ("constant_heating_rate", ChimesFloat), 
                ("abundances", ct.POINTER(ChimesFloat)), 
                ("hybrid_data", ct.c_void_p)]

# Global Variables structure
class globalVariables(ct.Structure):
    pass 

globalVariables._fields_ = [("MainDataTablePath", ct.c_char * 500), 
                            ("PhotoIonTablePath", (ct.c_char * 500) * 20), 
                            ("EqAbundanceTablePath", ct.c_char * 500), 
                            ("cellSelfShieldingOn", ct.c_int), 
                            ("N_spectra", ct.c_int), 
                            ("rt_update_flux", ct.c_int), 
                            ("rt_use_on_the_spot_approx", ct.c_int), 
                            ("redshift_dependent_UVB_index", ct.c_int), 
                            ("use_redshift_dependent_eqm_tables", ct.c_int), 
                            ("redshift", ChimesFloat), 
                            ("reionisation_redshift", ChimesFloat), 
                            ("StaticMolCooling", ct.c_int), 
                            ("Tmol_K", ChimesFloat), 
                            ("grain_temperature", ChimesFloat), 
                            ("cmb_temperature", ChimesFloat), 
                            ("relativeTolerance", ChimesFloat), 
                            ("absoluteTolerance", ChimesFloat), 
                            ("explicitTolerance", ChimesFloat), 
                            ("rt_density_absoluteTolerance", ChimesFloat), 
                            ("rt_flux_absoluteTolerance", ChimesFloat), 
                            ("element_included", ct.c_int * 9),
                            ("speciesIndices", ct.c_int * 157), 
                            ("totalNumberOfSpecies", ct.c_int), 
                            ("scale_metal_tolerances", ct.c_int), 
                            ("chimes_debug", ct.c_int), 
                            ("hybrid_cooling_mode", ct.c_int), 
                            ("hybrid_data", ct.c_void_p),  
                            ("hybrid_cooling_fn", ct.CFUNCTYPE(ct.POINTER(gasVariables), ct.POINTER(globalVariables))),   
                            ("allocate_gas_hybrid_data_fn", ct.CFUNCTYPE(ct.POINTER(gasVariables))), 
                            ("free_gas_hybrid_data_fn", ct.CFUNCTYPE(ct.POINTER(gasVariables)))] 

class chimes_spectra_struct(ct.Structure): 
    _fields_ = [("isotropic_photon_density", ct.POINTER(ChimesFloat)), 
                ("G0_parameter", ct.POINTER(ChimesFloat)), 
                ("H2_dissocJ", ct.POINTER(ChimesFloat)), 
                ("HII_recombination_flag", ct.POINTER(ct.c_int)), 
                ("HeII_recombination_flag", ct.POINTER(ct.c_int))] 

class chimes_current_rates_struct(ct.Structure): 
    _fields_ = [("data_buffer", ct.POINTER(ChimesFloat)), 
                ("T_dependent_rate_coefficient", ct.POINTER(ChimesFloat)), 
                ("T_dependent_rate", ct.POINTER(ChimesFloat)), 
                ("constant_rate", ct.POINTER(ChimesFloat)), 
                ("recombination_AB_rate_coefficient", ct.POINTER(ChimesFloat)), 
                ("recombination_AB_rate", ct.POINTER(ChimesFloat)), 
                ("grain_recombination_rate_coefficient", ct.POINTER(ChimesFloat)), 
                ("grain_recombination_rate", ct.POINTER(ChimesFloat)), 
                ("cosmic_ray_rate", ct.POINTER(ChimesFloat)), 
                ("CO_cosmic_ray_rate_coefficient", ct.POINTER(ChimesFloat)), 
                ("CO_cosmic_ray_rate", ct.POINTER(ChimesFloat)), 
                ("H2_dust_formation_rate_coefficient", ChimesFloat), 
                ("H2_dust_formation_rate", ChimesFloat), 
                ("H2_collis_dissoc_rate_coefficient", ct.POINTER(ChimesFloat)), 
                ("H2_collis_dissoc_rate", ct.POINTER(ChimesFloat)), 
                ("H2_collis_dissoc_crit_H", ChimesFloat), 
                ("H2_collis_dissoc_crit_H2", ChimesFloat), 
                ("H2_collis_dissoc_crit_He", ChimesFloat), 
                ("H2_collis_dissoc_log_k0", ct.POINTER(ChimesFloat)), 
                ("H2_collis_dissoc_log_kLTE", ct.POINTER(ChimesFloat)), 
                ("photoion_fuv_shield_factor", ct.POINTER(ChimesFloat)), 
                ("photoion_fuv_rate_coefficient", ct.POINTER(ChimesFloat)), 
                ("photoion_fuv_rate", ct.POINTER(ChimesFloat)), 
                ("photoion_fuv_heat_rate", ct.POINTER(ChimesFloat)), 
                ("photoion_euv_shield_factor", ct.POINTER(ChimesFloat)), 
                ("photoion_euv_rate_coefficient", ct.POINTER(ChimesFloat)), 
                ("photoion_euv_rate", ct.POINTER(ChimesFloat)), 
                ("photoion_euv_epsilon", ct.POINTER(ChimesFloat)), 
                ("photoion_euv_heat_rate", ct.POINTER(ChimesFloat)), 
                ("photoion_auger_fuv_rate_coefficient", ct.POINTER(ChimesFloat)), 
                ("photoion_auger_fuv_rate", ct.POINTER(ChimesFloat)), 
                ("photoion_auger_euv_rate_coefficient", ct.POINTER(ChimesFloat)), 
                ("photoion_auger_euv_rate", ct.POINTER(ChimesFloat)), 
                ("photodissoc_group1_shield_factor", ct.POINTER(ChimesFloat)), 
                ("photodissoc_group1_rate_coefficient", ct.POINTER(ChimesFloat)), 
                ("photodissoc_group1_rate", ct.POINTER(ChimesFloat)), 
                ("photodissoc_group2_shield_factor", ChimesFloat), 
                ("photodissoc_group2_rate_coefficient", ct.POINTER(ChimesFloat)), 
                ("photodissoc_group2_rate", ct.POINTER(ChimesFloat)), 
                ("H2_photodissoc_shield_factor", ct.POINTER(ChimesFloat)), 
                ("H2_photodissoc_rate_coefficient", ct.POINTER(ChimesFloat)), 
                ("H2_photodissoc_rate", ct.POINTER(ChimesFloat)), 
                ("CO_photodissoc_shield_factor", ct.POINTER(ChimesFloat)), 
                ("CO_photodissoc_rate_coefficient", ct.POINTER(ChimesFloat)), 
                ("CO_photodissoc_rate", ct.POINTER(ChimesFloat)), 
                ("photon_density_rate", ct.POINTER(ChimesFloat)), 
                ("photon_flux_rate", ct.POINTER(ChimesFloat)), 
                ("cooling_rate", ct.POINTER(ChimesFloat)), 
                ("cooling_rate_2d", ct.POINTER(ChimesFloat)), 
                ("cooling_rate_4d", ct.POINTER(ChimesFloat))] 

class Species_Structure(ct.Structure): 
    _fields_ = [("include_species", ct.c_int), 
                ("element_abundance", ChimesFloat), 
                ("creation_rate", ChimesFloat), 
                ("destruction_rate", ChimesFloat)] 

class UserData(ct.Structure): 
    _fields_ = [("myGasVars", ct.POINTER(gasVariables)), 
                ("myGlobalVars", ct.POINTER(globalVariables)), 
                ("species", ct.POINTER(Species_Structure)), 
                ("chimes_current_rates", ct.POINTER(chimes_current_rates_struct)), 
                ("cvode_mem", ct.c_void_p), 
                ("HI_column", ChimesFloat), 
                ("H2_column", ChimesFloat), 
                ("HeI_column", ChimesFloat), 
                ("HeII_column", ChimesFloat), 
                ("CO_column", ChimesFloat), 
                ("H2O_column", ChimesFloat), 
                ("OH_column", ChimesFloat), 
                ("extinction", ChimesFloat), 
                ("network_size", ct.c_int), 
                ("mol_flag_index", ct.c_int), 
                ("case_AB_index", ct.c_int * 2)] 
                
## set global variables in ctypes global variables structure
def initializeDefaultGlobalVariables(
    myGlobalVars,
    driver_parameters,
    global_variable_parameters): 

    ## Paths set according to the default chimes-data directory structure 
    myGlobalVars.MainDataTablePath = (
        "%s/chimes_main_data.hdf5" % (driver_parameters["chimes_data_path"])
        ).encode('UTF-8')
    if driver_parameters["EqAbundanceTable_filename"] == None: 
        myGlobalVars.EqAbundanceTablePath = ("None").encode('UTF-8')
    else: 
        myGlobalVars.EqAbundanceTablePath = ("%s/EqAbundancesTables/%s" % (
            driver_parameters["chimes_data_path"],
            driver_parameters["EqAbundanceTable_filename"])
        ).encode('UTF-8')

    # Redshift dependent UVB is off 
    # by default. We will switch it 
    # on if needed. 
    myGlobalVars.redshift_dependent_UVB_index = -1 

    if driver_parameters["UV_field"] == None:  
        myGlobalVars.N_spectra = 0 

    elif driver_parameters["UV_field"] == "HM01":
        myGlobalVars.N_spectra = 1 

        # NOTE: HM01 only has the z0.000 
        # cross sections file in chimes-data, 
        # so we cannot yet use the redshift-
        # dependent UVB option here. 
        photoion_string_list = list("%s/HM01_cross_sections/z%.3f_cross_sections.hdf5" % (driver_parameters["chimes_data_path"], global_variable_parameters["redshift"])) 
        for i in range(len(photoion_string_list)): 
            myGlobalVars.PhotoIonTablePath[0][i] = photoion_string_list[i].encode('UTF-8')  

    elif driver_parameters["UV_field"] == "HM12":
        myGlobalVars.N_spectra = 1 
        myGlobalVars.redshift_dependent_UVB_index = 0 

        # When using the redshift-dependent UVB 
        # option, we only specify the path to 
        # directory containing the cross sections 
        # files, not the file path itself. 
        photoion_string_list = list("%s/HM12_cross_sections" % (driver_parameters["chimes_data_path"], )) 
        for i in range(len(photoion_string_list)): 
            myGlobalVars.PhotoIonTablePath[0][i] = photoion_string_list[i].encode('UTF-8')  

    elif driver_parameters["UV_field"] == "FG20":
        myGlobalVars.N_spectra = 1 
        myGlobalVars.redshift_dependent_UVB_index = 0 
        
        # When using the redshift-dependent UVB 
        # option, we only specify the path to 
        # directory containing the cross sections 
        # files, not the file path itself. 
        photoion_string_list = list("%s/FG20_cross_sections" % (driver_parameters["chimes_data_path"], )) 
        for i in range(len(photoion_string_list)): 
            myGlobalVars.PhotoIonTablePath[0][i] = photoion_string_list[i].encode('UTF-8')  

    elif driver_parameters["UV_field"] == "B87": 
        myGlobalVars.N_spectra = 1 
        photoion_string_list = list("%s/cross_sections_B87.hdf5" % (driver_parameters["chimes_data_path"])) 
        for i in range(len(photoion_string_list)): 
            myGlobalVars.PhotoIonTablePath[0][i] = photoion_string_list[i].encode('UTF-8')  

    elif driver_parameters["UV_field"] == "Colibre": 
        myGlobalVars.N_spectra = 2 

        myGlobalVars.redshift_dependent_UVB_index = 0 
        photoion_string_list = list("%s/SP20_cross_sections" % (driver_parameters["chimes_data_path"], )) 
        for i in range(len(photoion_string_list)): 
            myGlobalVars.PhotoIonTablePath[0][i] = photoion_string_list[i].encode('UTF-8')  

        photoion_string_list = list("%s/cross_sections_B87.hdf5" % (driver_parameters["chimes_data_path"])) 
        for i in range(len(photoion_string_list)): 
            myGlobalVars.PhotoIonTablePath[1][i] = photoion_string_list[i].encode('UTF-8')  

    elif driver_parameters["UV_field"] == "StellarFluxes": 
        myGlobalVars.N_spectra = 9 

        myGlobalVars.redshift_dependent_UVB_index = 0 
        photoion_string_list = list("%s/FG20_cross_sections" % (driver_parameters["chimes_data_path"], )) 
        for i in range(len(photoion_string_list)): 
            myGlobalVars.PhotoIonTablePath[0][i] = photoion_string_list[i].encode('UTF-8')  

        for i in range(8): 
            photoion_string_list = list("%s/starburstCrossSections/cross_sections_SB%d.hdf5" % (driver_parameters["chimes_data_path"], i + 1)) 
            for j in range(len(photoion_string_list)): 
                myGlobalVars.PhotoIonTablePath[i + 1][j] = photoion_string_list[j].encode('UTF-8')
                
    elif driver_parameters["UV_field"] == "S04": 
        myGlobalVars.N_spectra = 1 
        photoion_string_list = list("%s/cross_sections_S04.hdf5" % (driver_parameters["chimes_data_path"])) 
        for i in range(len(photoion_string_list)): 
            myGlobalVars.PhotoIonTablePath[0][i] = photoion_string_list[i].encode('UTF-8')  

    else: 
        raise KeyError("UV_field %s not recognised. Aborting." % (UV_field, ) )
    
    ## Set cellSelfShieldingOn from shield_mode 
    if driver_parameters["shield_mode"] == None: 
        # No self shielding 
        myGlobalVars.cellSelfShieldingOn = 0
    elif (driver_parameters["shield_mode"] == "read-in" or 
        driver_parameters["shield_mode"] == "Jeans" or 
        driver_parameters["shield_mode"] == "Colibre"): 
	# Self shielding, but column densities of individual 
	# species are not updated during the hydro time-step. 
        myGlobalVars.cellSelfShieldingOn = 1 
    else: 
        raise KeyError(
            "shield_mode %d not recognised. Aborting." % (
                driver_parameters["shield_mode"]))

    # The remaining parameters are read in from the parameter file
    # and stored in global_variable_parameters
    for key in global_variable_parameters:
        setattr(myGlobalVars, key, global_variable_parameters[key]) 
    
def initializeDefaultGasVariables(myGasVars, gas_variable_parameters):
    for key in gas_variable_parameters:
        setattr(myGasVars, key, gas_variable_parameters[key])
