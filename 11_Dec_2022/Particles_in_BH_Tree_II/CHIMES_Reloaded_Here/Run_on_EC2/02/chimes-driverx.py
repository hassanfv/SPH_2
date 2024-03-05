## import from standard libraries
import numpy as np
import ctypes as ct
import sys
import h5py
import copy
import time
import os

try: 
    ## import chimes library ctype structs and their initialization functions
    from utils.chimes_classes import *

    ## import relevant helper functions
    from utils.snapshot_utils import SnapshotData, snapshot_check_output_arrays 
    from utils.grid_utils import create_grid, grid_check_output_arrays 
    from utils.chimes_utils import compute_cooling_rates 
    from utils.shielding_utils import compute_jeans_shield_length, compute_colibre_Nref, compute_colibre_shield_length 
    from utils.dust_utils import J09DC16_set_depletion_factors, colibre_set_depletion_factors 
    from utils.uv_field_utils import compute_stellar_fluxes, compute_colibre_ISRF, compute_AGN_isotropic_photon_density 
    from utils.phys_const import solar_mass_cgs 
except ImportError:
    ## On some Python distributions it cannot
    ## find the above modules in utils. We
    ## instead have to manually add the utils
    ## directory to the path and import from there. 
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
    
    ## import chimes library ctype structs and their initialization functions
    from chimes_classes import *

    ## import relevant helper functions
    from snapshot_utils import SnapshotData, snapshot_check_output_arrays 
    from grid_utils import create_grid, grid_check_output_arrays 
    from chimes_utils import compute_cooling_rates 
    from shielding_utils import compute_jeans_shield_length, compute_colibre_Nref, compute_colibre_shield_length 
    from dust_utils import J09DC16_set_depletion_factors, colibre_set_depletion_factors 
    from uv_field_utils import compute_stellar_fluxes, compute_colibre_ISRF, compute_AGN_isotropic_photon_density 
    from phys_const import solar_mass_cgs 
    
## mpi parallelization
#from mpi4py import MPI

## import parameter routines 
from driver_config import read_parameters, print_parameters 

class ChimesDriver(object): 
    def __init__(self, data_list, driver_pars,
                 global_variable_pars, gas_variable_pars, rank):

        # Parameters 
        self.driver_pars = driver_pars
        self.global_variable_pars = global_variable_pars
        self.gas_variable_pars = gas_variable_pars 
        self.rank = rank 
        
        ## sph particle information
        self.nH_arr = data_list[0] 
        self.temperature_arr = data_list[1] 
        self.metallicity_arr = data_list[2] 
        self.shieldLength_arr = data_list[3] 
        self.init_chem_arr = data_list[4] 
        if driver_pars["UV_field"] == "StellarFluxes": 
            self.ChimesFluxIon_arr = data_list[5] 
            self.ChimesFluxG0_arr = data_list[6]
        elif driver_pars["UV_field"] == "S04" and driver_pars["IO_mode"] == "snapshot":
            self.distance_to_AGN = data_list[5] 

        self.N_part = len(self.nH_arr) 
        
        self.init_chimesLib()
        self.init_chimes_constant_arg_structs()
        self.init_chimes_solver()
        self.init_myGasVars()

        ## initialize output arrays 
        if self.driver_pars["driver_mode"] == "eqm_state" or self.driver_pars["driver_mode"] == "eqm_table": 
            self.output_array = np.zeros((self.N_part, self.myGlobalVars.totalNumberOfSpecies), dtype = np.float64) 
        elif self.driver_pars["driver_mode"] == "cooling_rates": 
            self.output_array = np.zeros((self.N_part, 2), dtype = np.float64) 
        elif self.driver_pars["driver_mode"] == "noneq_evolution": 
            self.output_array = np.zeros((self.N_part, self.myGlobalVars.totalNumberOfSpecies + 1, self.driver_pars["n_iterations"] + 1), dtype = np.float64) 
        else: 
            raise Exception("ERROR: driver_mode %s not recognised. Aborting" % (driver_pars["driver_mode"], )) 

        # Record time in chemistry solver 
        self.chimes_cumulative_time = 0.0  # seconds 

        # This array will record the element 
        # indices for each ion/molecule species 
        # in the network. Only used for the 
        # eqm_table mode. 
        self.species_element_indices = np.ones(157, dtype = np.int64) * (-1)

    def init_chimesLib(self):
	# Initialise CHIMES within the __init__ routine, so that we only 
	# have one process at a time reading in the CHIMES tables (otherwise 
	# it can cause problems). 
	# NOTE: The paths to the CHIMES library and data files are hard-coded here. 
        self.chimesLib = ct.cdll.LoadLibrary(self.driver_pars["chimes_library_path"])

        # For each library function that we are going to call 
        # from Python, we need to define its argument types. 
        self.chimesLib.init_chimes.argtypes = [ct.POINTER(globalVariables)] 

        self.chimesLib.allocate_gas_abundances_memory.argtypes = [
            ct.POINTER(gasVariables), ct.POINTER(globalVariables)] 

        self.chimesLib.chimes_network.argtypes = [
            ct.POINTER(gasVariables), ct.POINTER(globalVariables), ct.c_int]

    def init_chimes_constant_arg_structs(self):
	## initialize new instances of the chimes ctypes structures for the variables
        self.myGlobalVars = globalVariables() 
        self.myGasVars = gasVariables() 
	
	## functions defined in chimes-classes.py
        initializeDefaultGlobalVariables(self.myGlobalVars, self.driver_pars, self.global_variable_pars)
        initializeDefaultGasVariables(self.myGasVars, self.gas_variable_pars)
    
    def init_chimes_solver(self):
        if self.rank == 0: 
            print("Initialising CHIMES...")
            sys.stdout.flush() 

        ## Currently, CHIMES Driver does not 
        ## support coupling CHIMES to RT. 
        ## Make sure this option is 
        ## switched off. 
        if self.myGlobalVars.rt_update_flux != 0: 
            raise Exception("ERROR: rt_update_flux has been set to %d. CHIMES-Driver does not currently support coupling CHIMES to RT, you must set this option to zero. Aborting." % (self.driver_pars["rt_update_flux"], )) 

	## main call to init_chimes
        self.chimesLib.init_chimes(self.myGlobalVars) 
        
        ## This structure contains the G0_parameter 
        ## and H2_dissocJ read in from the cross 
        ## sections tables within CHIMES. 
        self.spectra_table = chimes_spectra_struct.in_dll(self.chimesLib, "chimes_table_spectra") 

        print("Finished initialising rank %d" % (self.rank, ))
        sys.stdout.flush() 

    def init_myGasVars(self):
        # Allocate memory to the abundances
        # array in myGasVars
        self.chimesLib.allocate_gas_abundances_memory(self.myGasVars, self.myGlobalVars)

        # Set the UV flux. 
        if self.driver_pars["UV_field"] == None:
            # No radiation 
            return 
        elif self.driver_pars["UV_field"] == "HM01": 
            # Haardt & Madau (2001), redshift zero. 
            self.myGasVars.isotropic_photon_density[0] = self.spectra_table.isotropic_photon_density[0] * self.driver_pars["radiation_field_normalisation_factor"] 
            self.myGasVars.G0_parameter[0] = self.spectra_table.G0_parameter[0]  
            self.myGasVars.H2_dissocJ[0] = self.spectra_table.H2_dissocJ[0] 
        elif self.driver_pars["UV_field"] == "HM12": 
            # Haardt & Madau (2012), redshift-dependent. 
            self.myGasVars.isotropic_photon_density[0] = self.spectra_table.isotropic_photon_density[0] * self.driver_pars["radiation_field_normalisation_factor"] 
            self.myGasVars.G0_parameter[0] = self.spectra_table.G0_parameter[0]  
            self.myGasVars.H2_dissocJ[0] = self.spectra_table.H2_dissocJ[0] 
        elif self.driver_pars["UV_field"] == "FG20": 
            # Faucher-Giguere (2020), redshift-dependent.
            self.myGasVars.isotropic_photon_density[0] = self.spectra_table.isotropic_photon_density[0] * self.driver_pars["radiation_field_normalisation_factor"] 
            self.myGasVars.G0_parameter[0] = self.spectra_table.G0_parameter[0]  
            self.myGasVars.H2_dissocJ[0] = self.spectra_table.H2_dissocJ[0] 
        elif self.driver_pars["UV_field"] == "B87": 
            # Black (1987) 
            self.myGasVars.isotropic_photon_density[0] = self.spectra_table.isotropic_photon_density[0] * self.driver_pars["radiation_field_normalisation_factor"] 
            self.myGasVars.G0_parameter[0] = self.spectra_table.G0_parameter[0]  
            self.myGasVars.H2_dissocJ[0] = self.spectra_table.H2_dissocJ[0] 
        elif self.driver_pars["UV_field"] == "Colibre": 
            # SP20 redshift-dependent UVB plus B87 ISRF. 
            self.myGasVars.isotropic_photon_density[0] = self.spectra_table.isotropic_photon_density[0] 
            self.myGasVars.G0_parameter[0] = self.spectra_table.G0_parameter[0]  
            self.myGasVars.H2_dissocJ[0] = self.spectra_table.H2_dissocJ[0] 

            self.myGasVars.isotropic_photon_density[1] = self.spectra_table.isotropic_photon_density[1] 
            self.myGasVars.G0_parameter[1] = self.spectra_table.G0_parameter[1]  
            self.myGasVars.H2_dissocJ[1] = self.spectra_table.H2_dissocJ[1] 
        elif self.driver_pars["UV_field"] == "StellarFluxes": 
            # FG20 redshift-dependent UVB plus 
            # SB99 stellar flux spectra in 
            # 8 age bins 
            self.myGasVars.isotropic_photon_density[0] = self.spectra_table.isotropic_photon_density[0] 
            self.myGasVars.G0_parameter[0] = self.spectra_table.G0_parameter[0] 
            self.myGasVars.H2_dissocJ[0] = self.spectra_table.H2_dissocJ[0] 
        elif self.driver_pars["UV_field"] == "S04": 
            # Average quasar spectrum 
            # from Sazonov et al. (2004)
            if self.driver_pars["IO_mode"] == "snapshot":
                # Set at 1 kpc for now. This will be
                # replaced for each particle later. 
                self.myGasVars.isotropic_photon_density[0] = compute_AGN_isotropic_photon_density(self.driver_pars["bolometric_AGN_luminosity_cgs"], 1.0)
            else: 
                self.myGasVars.isotropic_photon_density[0] = compute_AGN_isotropic_photon_density(self.driver_pars["bolometric_AGN_luminosity_cgs"], self.driver_pars["distance_to_AGN_kpc"])
            self.myGasVars.G0_parameter[0] = self.spectra_table.G0_parameter[0]  
            self.myGasVars.H2_dissocJ[0] = self.spectra_table.H2_dissocJ[0] 
        else: 
            raise KeyError("UV_field %s not recognised. Aborting." % (self.driver_pars["UV_field"], ))

        return 

    def set_species_element_indices(self): 
        # For each ion/molecule species, determine 
        # the index of its corresponding element. 
        # This is needed for the eqm_table mode, 
        # where we will output the abundances 
        # relative to the corresponding element 
        # abundances, rather than to hydrogen. 
        current_index = 0 
        
        # H species 
        for i in range(4): 
            self.species_element_indices[current_index] = -1 
            current_index += 1 
                
        # He species 
        for i in range(4, 7): 
            self.species_element_indices[current_index] = 0 
            current_index += 1 

        # C species 
        for i in range(7, 15): 
            if self.myGlobalVars.element_included[0] == 1: 
                self.species_element_indices[current_index] = 1 
                current_index += 1 

        # N species 
        for i in range(15, 23): 
            if self.myGlobalVars.element_included[1] == 1: 
                self.species_element_indices[current_index] = 2 
                current_index += 1 

        # O species 
        for i in range(23, 33): 
            if self.myGlobalVars.element_included[2] == 1: 
                self.species_element_indices[current_index] = 3 
                current_index += 1 

        # Ne species 
        for i in range(33, 44): 
            if self.myGlobalVars.element_included[3] == 1: 
                self.species_element_indices[current_index] = 4 
                current_index += 1 

        # Mg species 
        for i in range(44, 57): 
            if self.myGlobalVars.element_included[4] == 1: 
                self.species_element_indices[current_index] = 5 
                current_index += 1 

        # Si species 
        for i in range(57, 72): 
            if self.myGlobalVars.element_included[5] == 1: 
                self.species_element_indices[current_index] = 6 
                current_index += 1 

        # S species 
        for i in range(72, 89): 
            if self.myGlobalVars.element_included[6] == 1: 
                self.species_element_indices[current_index] = 7 
                current_index += 1 

        # Ca species 
        for i in range(89, 110): 
            if self.myGlobalVars.element_included[7] == 1: 
                self.species_element_indices[current_index] = 8 
                current_index += 1 

        # Fe species 
        for i in range(110, 137): 
            if self.myGlobalVars.element_included[8] == 1: 
                self.species_element_indices[current_index] = 9 
                current_index += 1 

        # H-molecules 
        for i in range(137, 140): 
            self.species_element_indices[current_index] = -1 
            current_index += 1 

        # OH 
        if self.myGlobalVars.element_included[2] == 1: 
            self.species_element_indices[current_index] = 3 
            current_index += 1 

        # H2O 
        if self.myGlobalVars.element_included[2] == 1: 
            self.species_element_indices[current_index] = 3 
            current_index += 1 

        # C2
        if self.myGlobalVars.element_included[0] == 1: 
            self.species_element_indices[current_index] = 1 
            current_index += 1 

        # O2
        if self.myGlobalVars.element_included[2] == 1: 
            self.species_element_indices[current_index] = 3 
            current_index += 1 

        # HCOp
        if self.myGlobalVars.element_included[0] == 1 and self.myGlobalVars.element_included[2] == 1: 
            self.species_element_indices[current_index] = 1 
            current_index += 1 

        # CH
        if self.myGlobalVars.element_included[0] == 1: 
            self.species_element_indices[current_index] = 1 
            current_index += 1 

        # CH2
        if self.myGlobalVars.element_included[0] == 1: 
            self.species_element_indices[current_index] = 1 
            current_index += 1 

        # CH3p
        if self.myGlobalVars.element_included[0] == 1: 
            self.species_element_indices[current_index] = 1 
            current_index += 1 

        # CO
        if self.myGlobalVars.element_included[0] == 1 and self.myGlobalVars.element_included[2] == 1: 
            self.species_element_indices[current_index] = 1 
            current_index += 1 

        # CHp
        if self.myGlobalVars.element_included[0] == 1: 
            self.species_element_indices[current_index] = 1 
            current_index += 1 

        # CH2p
        if self.myGlobalVars.element_included[0] == 1: 
            self.species_element_indices[current_index] = 1 
            current_index += 1 

        # OHp
        if self.myGlobalVars.element_included[2] == 1: 
            self.species_element_indices[current_index] = 3 
            current_index += 1 

        # H2Op
        if self.myGlobalVars.element_included[2] == 1: 
            self.species_element_indices[current_index] = 3 
            current_index += 1 

        # H3Op
        if self.myGlobalVars.element_included[2] == 1: 
            self.species_element_indices[current_index] = 3 
            current_index += 1 

        # COp
        if self.myGlobalVars.element_included[0] == 1 and self.myGlobalVars.element_included[2] == 1: 
            self.species_element_indices[current_index] = 1 
            current_index += 1 

        # HOCp
        if self.myGlobalVars.element_included[0] == 1 and self.myGlobalVars.element_included[2] == 1: 
            self.species_element_indices[current_index] = 1 
            current_index += 1 

        # O2p
        if self.myGlobalVars.element_included[2] == 1: 
            self.species_element_indices[current_index] = 3 
            current_index += 1 

        return 

    def run(self): 
        if self.driver_pars["driver_mode"] == "eqm_table": 
            self.set_species_element_indices() 

        # Loop through all particles 
        for i in range(self.N_part): 
            if self.rank == 0: 
                if i % 100 == 0: 
                    print("Rank %d: Particle %d of %d" % (self.rank, i, self.N_part))
                    sys.stdout.flush()

            self.myGasVars.cell_size = self.shieldLength_arr[i] 
            self.myGasVars.temperature = self.temperature_arr[i] 
            self.myGasVars.nH_tot = self.nH_arr[i] 
            self.myGasVars.metallicity = self.metallicity_arr[i, 0] / 0.0129 
            self.myGasVars.dust_ratio = self.myGasVars.metallicity 

            XH = 1.0 - (self.metallicity_arr[i, 0] + self.metallicity_arr[i, 1]) 

            if self.driver_pars["UV_field"] == "Colibre" or self.driver_pars["dust_depletion"] == "Colibre": 
                N_ref = compute_colibre_Nref(self.temperature_arr[i], self.nH_arr[i], XH, self.driver_pars["max_shield_length"], self.driver_pars["colibre_log_T_min"], self.driver_pars["colibre_log_T_max"], self.driver_pars["colibre_use_turbulent_jeans_length"], self.driver_pars["colibre_sigma_turb"])
                N_H0 = 3.63e20    # Milky Way value 

                if self.driver_pars["colibre_saturate_radiation_field"] == 1: 
                    N_scale_rad = min(N_ref, self.driver_pars["colibre_col_dens_saturate_rad"]) 
                else: 
                    N_scale_rad = N_ref 

                if self.driver_pars["colibre_saturate_cr_rate"] == 1: 
                    N_scale_cr = min(N_ref, self.driver_pars["colibre_col_dens_saturate_cr"]) 
                else: 
                    N_scale_cr = N_ref 

            if self.driver_pars["UV_field"] == "Colibre": 
                compute_colibre_ISRF(self.myGasVars, self.myGlobalVars, self.spectra_table, N_scale_rad / N_H0, N_scale_cr / N_H0, self.driver_pars["colibre_scale_MW_ISRF"], self.gas_variable_pars["cr_rate"], self.driver_pars["colibre_ISRF_low_dens_cut_off_redshift"], self.driver_pars["colibre_cr_plaw_index"]) 
            elif self.driver_pars["UV_field"] == "StellarFluxes": 
                # UVB is constant, but stellar fluxes 
                # need to be set for each particle. 
                for j in range(8): 
                    self.myGasVars.isotropic_photon_density[j + 1] = self.ChimesFluxIon_arr[i, j] / 3.0e10 
                    self.myGasVars.G0_parameter[j + 1] = self.ChimesFluxG0_arr[i, j] / max(self.ChimesFluxIon_arr[i, j], 1.0e-100) 
                    self.myGasVars.H2_dissocJ[j + 1] = self.myGasVars.G0_parameter[j + 1] * self.spectra_table.H2_dissocJ[j + 1] / self.spectra_table.G0_parameter[j + 1]
            elif self.driver_pars["UV_field"] == "S04" and self.driver_pars["IO_mode"] == "snapshot":
                self.myGasVars.isotropic_photon_density[0] = compute_AGN_isotropic_photon_density(self.driver_pars["bolometric_AGN_luminosity_cgs"], self.distance_to_AGN[i])
                
            atomic_masses = np.array([4., 12., 14., 16., 20., 24., 28., 32., 40., 56.])
            metal_number_abundances = self.metallicity_arr[i, 1:11] / (atomic_masses * XH)
            self.myGasVars.element_abundances = (ChimesFloat * 10)(
		metal_number_abundances[0],metal_number_abundances[1],metal_number_abundances[2],
                metal_number_abundances[3],metal_number_abundances[4],metal_number_abundances[5],
                metal_number_abundances[6],metal_number_abundances[7],metal_number_abundances[8],
                metal_number_abundances[9])

            if self.driver_pars["dust_depletion"] == "J09" or self.driver_pars["dust_depletion"] == "DC16": 
                # Deplete metals based on the Jenkins (2009) 
                # and/or De Cia et al (2016) depletion factors. 
                # See Richings et al. (in prep) for details of 
                # how this is implemented as a function of 
                # gas density. 
                J09DC16_set_depletion_factors(self.myGasVars, self.driver_pars["dust_depletion"]) 
            elif self.driver_pars["dust_depletion"] == "Colibre": 
                # Deplete metals as in Colibre 
                colibre_set_depletion_factors(self.myGasVars, N_scale_rad / N_H0) 
            else: 
                if self.driver_pars["dust_depletion"] != None: 
                    raise Exception("ERROR: dust_depletion %s not recognised." % (self.driver_pars["dust_depletion"], )) 

            if self.driver_pars["dust_boost_mode"] == 0: 
                # Do not boost reactions on dust grains. 
                self.myGasVars.dust_boost_factor = 1.0 
            elif self.driver_pars["dust_boost_mode"] == 1: 
                # Apply a constant dust boost factor. 
                self.myGasVars.dust_boost_factor = self.driver_pars["dust_boost_factor"] 
            elif self.driver_pars["dust_boost_mode"] == 2: 
                # Apply a density-dependent dust boost factor. 
                if self.myGasVars.nH_tot <= self.driver_pars["dust_boost_nH_min_cgs"]: 
                    self.myGasVars.dust_boost_factor = 1.0 
                elif self.myGasVars.nH_tot >= self.driver_pars["dust_boost_nH_max_cgs"]: 
                    self.myGasVars.dust_boost_factor = self.driver_pars["dust_boost_factor"] 
                else: 
                    log_nH_particle = np.log10(self.myGasVars.nH_tot) 
                    log_nH_min = np.log10(self.driver_pars["dust_boost_nH_min_cgs"])
                    log_nH_max = np.log10(self.driver_pars["dust_boost_nH_max_cgs"])
                    boost_max = self.driver_pars["dust_boost_factor"]
                    self.myGasVars.dust_boost_factor = np.power(10., ((log_nH_particle - log_nH_min) / max((log_nH_max - log_nH_min), 1.0e-38)) * np.log10(boost_max))
            else: 
                raise Exception("ERROR: dust_boost_mode %d not recognised. Allowed options are: 0 (not dust boost factor), 1 (constant dust boost factor), or 2 (density-dependent dust boost factor)." % (self.driver_pars["dust_boost_mode"], )) 

            # Copy over the initial chemical state. 
            for j in range(self.myGlobalVars.totalNumberOfSpecies):
                self.myGasVars.abundances[j] = self.init_chem_arr[i, j] 

            if self.driver_pars["driver_mode"] == "noneq_evolution": 
                # First, write the initial abundances and temperature 
                # to the output array 
                for j in range(self.myGlobalVars.totalNumberOfSpecies): 
                    self.output_array[i, j, 0] = self.myGasVars.abundances[j] 
                self.output_array[i, self.myGlobalVars.totalNumberOfSpecies, 0] = self.myGasVars.temperature 

                # Run the chemistry solver n_iterations times, 
                # recording the abundances and temperature after 
                # each call. Note that if ThermEvolOn == 0, the 
                # temperature will be constant. 
                for k in range(self.driver_pars["n_iterations"]):
                    t1 = time.time() 
                    self.chimesLib.chimes_network(self.myGasVars, self.myGlobalVars, 0)
                    t2 = time.time() 
                    self.chimes_cumulative_time += (t2 - t1) 
                
                    for j in range(self.myGlobalVars.totalNumberOfSpecies): 
                        self.output_array[i, j, k + 1] = self.myGasVars.abundances[j] 
                    self.output_array[i, self.myGlobalVars.totalNumberOfSpecies, k + 1] = self.myGasVars.temperature 

                    # For COLIBRE UV field and depletion, we need to 
                    # update these after EVERY iteration. Also if 
                    # shield_mode == Jeans. Only needed if ThermEvolOn == 1 
                    if self.myGasVars.ThermEvolOn == 1: 
                        if self.driver_pars["UV_field"] == "Colibre" or self.driver_pars["shield_mode"] == "Colibre" or self.driver_pars["dust_depletion"] == "Colibre": 
                            N_ref = compute_colibre_Nref(self.myGasVars.temperature, self.myGasVars.nH_tot, XH, self.driver_pars["max_shield_length"], self.driver_pars["colibre_log_T_min"], self.driver_pars["colibre_log_T_max"], self.driver_pars["colibre_use_turbulent_jeans_length"], self.driver_pars["colibre_sigma_turb"])

                            if self.driver_pars["colibre_saturate_radiation_field"] == 1: 
                                N_scale_rad = min(N_ref, self.driver_pars["colibre_col_dens_saturate_rad"]) 
                            else: 
                                N_scale_rad = N_ref 

                            if self.driver_pars["colibre_saturate_cr_rate"] == 1: 
                                N_scale_cr = min(N_ref, self.driver_pars["colibre_col_dens_saturate_cr"]) 
                            else: 
                                N_scale_cr = N_ref 

                            if self.driver_pars["UV_field"] == "Colibre": 
                                compute_colibre_ISRF(self.myGasVars, self.myGlobalVars, self.spectra_table, N_scale_rad / N_H0, N_scale_cr / N_H0, self.driver_pars["colibre_scale_MW_ISRF"], self.gas_variable_pars["cr_rate"], self.driver_pars["colibre_ISRF_low_dens_cut_off_redshift"], self.driver_pars["colibre_cr_plaw_index"]) 

                            if self.driver_pars["shield_mode"] == "Colibre": 
                                self.myGasVars.cell_size = self.driver_pars["shield_length_factor"] * N_ref / self.myGasVars.nH_tot 

                            if self.driver_pars["dust_depletion"] == "Colibre": 
                                for idx in range(10): 
                                    self.myGasVars.element_abundances[idx] = metal_number_abundances[idx] 
                            
                                # Deplete metals as in Colibre 
                                colibre_set_depletion_factors(self.myGasVars, N_scale_rad / N_H0) 

                        if self.driver_pars["shield_mode"] == "Jeans": 
                            self.myGasVars.cell_size = compute_jeans_shield_length(self.myGasVars.temperature, self.myGasVars.nH_tot, self.driver_pars["shield_length_factor"], self.driver_pars["max_shield_length"])
            else: 
                # For driver_mode == eqm_state, eqm_table or cooling_rates, 
                # we still run the chemistry solver n_iterations times. This 
                # is especially important if shielding is on, as the 
                # column densities will update with each iteration. We 
                # then just record the final state. 
                for k in range(self.driver_pars["n_iterations"]):
                    t1 = time.time() 
                    self.chimesLib.chimes_network(self.myGasVars, self.myGlobalVars, 0)
                    t2 = time.time() 
                    self.chimes_cumulative_time += (t2 - t1) 

                    # For COLIBRE UV field and depletion, we need to 
                    # update these after EVERY iteration. Also if 
                    # shield_mode == Jeans. Only needed ThermEvolOn == 1 
                    if self.myGasVars.ThermEvolOn == 1: 
                        if self.driver_pars["UV_field"] == "Colibre" or self.driver_pars["shield_mode"] == "Colibre" or self.driver_pars["dust_depletion"] == "Colibre": 
                            N_ref = compute_colibre_Nref(self.myGasVars.temperature, self.myGasVars.nH_tot, XH, self.driver_pars["max_shield_length"], self.driver_pars["colibre_log_T_min"], self.driver_pars["colibre_log_T_max"], self.driver_pars["colibre_use_turbulent_jeans_length"], self.driver_pars["colibre_sigma_turb"])

                            if self.driver_pars["colibre_saturate_radiation_field"] == 1: 
                                N_scale_rad = min(N_ref, self.driver_pars["colibre_col_dens_saturate_rad"]) 
                            else: 
                                N_scale_rad = N_ref 

                            if self.driver_pars["colibre_saturate_cr_rate"] == 1: 
                                N_scale_cr = min(N_ref, self.driver_pars["colibre_col_dens_saturate_cr"]) 
                            else: 
                                N_scale_cr = N_ref 
                            
                            if self.driver_pars["UV_field"] == "Colibre": 
                                compute_colibre_ISRF(self.myGasVars, self.myGlobalVars, self.spectra_table, N_scale_rad / N_H0, N_scale_cr / N_H0, self.driver_pars["colibre_scale_MW_ISRF"], self.gas_variable_pars["cr_rate"], self.driver_pars["colibre_ISRF_low_dens_cut_off_redshift"], self.driver_pars["colibre_cr_plaw_index"]) 

                            if self.driver_pars["shield_mode"] == "Colibre": 
                                self.myGasVars.cell_size = self.driver_pars["shield_length_factor"] * N_ref / self.myGasVars.nH_tot 

                            if self.driver_pars["dust_depletion"] == "Colibre": 
                                for idx in range(10): 
                                    self.myGasVars.element_abundances[idx] = metal_number_abundances[idx] 
                            
                                # Deplete metals as in Colibre 
                                colibre_set_depletion_factors(self.myGasVars, N_scale_rad / N_H0) 

                        if self.driver_pars["shield_mode"] == "Jeans": 
                            self.myGasVars.cell_size = compute_jeans_shield_length(self.myGasVars.temperature, self.myGasVars.nH_tot, self.driver_pars["shield_length_factor"], self.driver_pars["max_shield_length"])

                # Write to output array 
                if self.driver_pars["driver_mode"] == "eqm_state": 
                    for j in range(self.myGlobalVars.totalNumberOfSpecies): 
                        self.output_array[i, j] = self.myGasVars.abundances[j] 
                        
                elif self.driver_pars["driver_mode"] == "eqm_table": 
                    # For the eqm tables, we record the abundances 
                    # relative to the corresponding element, and 
                    # not relative to hydrogen 
                    for j in range(self.myGlobalVars.totalNumberOfSpecies): 
                        element_index = self.species_element_indices[j] 
                        if element_index >= 0: 
                            self.output_array[i, j] = self.myGasVars.abundances[j] / max(self.myGasVars.element_abundances[element_index], 1.0e-38) 
                        else: 
                            self.output_array[i, j] = self.myGasVars.abundances[j]
                            
                elif self.driver_pars["driver_mode"] == "cooling_rates": 
                    lambda_cool, lambda_heat = compute_cooling_rates(self.myGasVars, self.myGlobalVars, self.chimesLib)
                    self.output_array[i, 0] = lambda_cool 
                    self.output_array[i, 1] = lambda_heat 

        print("Rank %d finished" % (self.rank, ))
        sys.stdout.flush()
        return self.output_array, self.chimes_cumulative_time 

def main():
    ## setup MPI variables
    #comm = MPI.COMM_WORLD
    rank = 0#comm.Get_rank() 
    N_task = 1#comm.Get_size()
    
    ## Parse parameter file 
    parameter_file = sys.argv[1]
    driver_pars, global_variable_pars, gas_variable_pars = read_parameters(parameter_file)    
    if rank == 0: 
        print_parameters(parameter_file, driver_pars, global_variable_pars, gas_variable_pars)
        if ChimesFloat == ct.c_float: 
            print("Using CHIMES in Single Precision")
        elif ChimesFloat == ct.c_double: 
            print("Using CHIMES in Double Precision")
        else:
            raise Exception("ERROR: ChimesFloat type not recognised. Aborting") 
        print(" ")
        sys.stdout.flush()
    
    if driver_pars["IO_mode"] == "snapshot": 
        # Check that the driver_mode is compatible 
        # with this IO_mode 
        if driver_pars["driver_mode"] != "eqm_state" and \
           driver_pars["driver_mode"] != "cooling_rates" and \
           driver_pars["driver_mode"] != "noneq_evolution": 
           raise Exception("ERROR: IO_mode %s cannot be run with driver_mode %s. Aborting" % (driver_pars["IO_mode"], driver_pars["driver_mode"])) 

        if rank == 0:
            # Check that output arrays don't already 
            # exist in output file. 
            output_check = snapshot_check_output_arrays(driver_pars) 
            if output_check == True: 
                raise Exception("ERROR: output array(s) already present in the output file. Aborting.") 

            # read arrays from snapshot
            my_snapshot_data = SnapshotData(driver_pars, global_variable_pars, gas_variable_pars) 
            
            if driver_pars["snapshot_type"] == "GIZMO": 
                my_snapshot_data.load_GIZMO() 
            elif driver_pars["snapshot_type"] == "AREPO": 
                my_snapshot_data.load_AREPO() 
            elif driver_pars["snapshot_type"] == "USER": 
                my_snapshot_data.load_USER() 
            else: 
                raise Exception("ERROR: snapshot type $s not recognised. Aborting" % (driver_pars["snapshot_type"], )) 
            
            nH_arr = my_snapshot_data.nH_arr 
            temperature_arr = my_snapshot_data.temperature_arr
            metallicity_arr = my_snapshot_data.metallicity_arr
            shieldLength_arr = my_snapshot_data.shieldLength_arr
            init_chem_arr = my_snapshot_data.init_chem_arr

            N_star = None 

            if driver_pars["UV_field"] == "StellarFluxes": 
                if driver_pars["compute_stellar_fluxes"] == 0: 
                    ChimesFluxIon_arr = my_snapshot_data.ChimesFluxIon_arr
                    ChimesFluxG0_arr = my_snapshot_data.ChimesFluxG0_arr
                else: 
                    gas_coords_arr = my_snapshot_data.gas_coords_arr 
                    star_coords_arr = my_snapshot_data.star_coords_arr 
                    star_mass_arr = my_snapshot_data.star_mass_arr 
                    star_age_Myr_arr = my_snapshot_data.star_age_Myr_arr 
                    N_star = len(star_mass_arr)
            elif driver_pars["UV_field"] == "S04": 
                gas_coords_arr = my_snapshot_data.gas_coords_arr 

            if driver_pars["driver_mode"] == "noneq_evolution": 
                # Save copies of the full temperature and 
                # density arrays to write out in the output 
                # file at the end. nH_arr and temperature_arr 
                # themselves will later be overwritten with 
                # only rank 0's own particles. 
                initial_temperature_array = temperature_arr.copy() 
                density_array = nH_arr.copy() 
            
            ## total number of particles
            N_tot = nH_arr.shape[0]

        else:
            N_tot = None
            N_star = None 
    elif driver_pars["IO_mode"] == "grid": 
        # Check that the driver_mode is compatible 
        # with this IO_mode 
        if driver_pars["driver_mode"] != "eqm_state" and \
           driver_pars["driver_mode"] != "eqm_table" and \
           driver_pars["driver_mode"] != "cooling_rates" and \
           driver_pars["driver_mode"] != "noneq_evolution": 
           raise Exception("ERROR: IO_mode %s cannot be run with driver_mode %s. Aborting" % (driver_pars["IO_mode"], driver_pars["driver_mode"])) 

        if rank == 0: 
            # Check that output arrays don't already 
            # exist in output file. 
            output_check = grid_check_output_arrays(driver_pars) 
            if output_check == True: 
                raise Exception("ERROR: output array(s) already present in the output file. Aborting.") 

            nH_arr, temperature_arr, metallicity_arr, shieldLength_arr, init_chem_arr = create_grid(driver_pars, global_variable_pars, gas_variable_pars["InitIonState"]) 
            if (driver_pars["UV_field"] == "StellarFluxes"): 
                raise Exception("ERROR: UV_field StellarFluxes is not compatible with IO_mode %s" % (driver_pars["IO_mode"], )) 

            ## total number of particles
            N_tot = nH_arr.shape[0]

        else: 
            N_tot = None 
    else: 
        raise KeyError("IO_mode %s not recognised. Aborting." % (driver_pars["IO_mode"], )) 

    ## send the total number of particles to each MPI task
    #N_tot = comm.bcast(N_tot,root=0)

    #if driver_pars["UV_field"] == "StellarFluxes" and driver_pars["compute_stellar_fluxes"] == 1: 
        #N_star = comm.bcast(N_star, root = 0) 

    ## split the particles among the tasks
    N_parts = [int(N_tot/N_task)+(i<N_tot%N_task) for i in range(N_task)]

    ## make sure we decomposed the arrays into chunks correctly
    if rank == 0: 
        print("Number of particles on each MPI task:") 
        print(N_parts)
        print (" ")
        sys.stdout.flush()

    assert sum(N_parts) == N_tot
    #comm.Barrier()

    ## allocate space for receiving the copy of the array chunks
    if rank !=0:
        nH_arr, temperature_arr, metallicity_arr, shieldLength_arr, init_chem_arr = [
            np.empty(N_parts[rank]),
            np.empty(N_parts[rank]),
            np.empty(N_parts[rank]),
            np.empty(N_parts[rank]),
            np.empty(N_parts[rank])]

        if driver_pars["UV_field"] == "StellarFluxes": 
            if driver_pars["compute_stellar_fluxes"] == 0: 
                ChimesFluxIon_arr = np.empty(N_parts[rank]) 
                ChimesFluxG0_arr = np.empty(N_parts[rank]) 
            else: 
                gas_coords_arr = np.empty(N_parts[rank]) 
                star_coords_arr = np.empty(N_star) 
                star_mass_arr = np.empty(N_star) 
                star_age_Myr_arr = np.empty(N_star)
        elif driver_pars["UV_field"] == "S04" and driver_pars["IO_mode"] == "snapshot": 
            gas_coords_arr = np.empty(N_parts[rank]) 

    ## send the particle array chunks to the child MPI tasks
    if rank == 0:
        current_left_index = N_parts[0]

        send_buffer = [nH_arr,temperature_arr,metallicity_arr,
                       shieldLength_arr,init_chem_arr]

        if driver_pars["UV_field"] == "StellarFluxes": 
            if driver_pars["compute_stellar_fluxes"] == 0: 
                send_buffer.append(ChimesFluxIon_arr) 
                send_buffer.append(ChimesFluxG0_arr) 
            else: 
                send_buffer.append(gas_coords_arr)
        elif driver_pars["UV_field"] == "S04" and driver_pars["IO_mode"] == "snapshot": 
            send_buffer.append(gas_coords_arr)

        for i in range(1,N_task):
            N_this_chunk = N_parts[i]

            #for j,arr in enumerate(send_buffer):

                ## send the array to the correct destination
                #comm.send(
                    ## copy the array because numpy masks arrays when you slice
                    #copy.copy(arr[current_left_index:current_left_index+N_this_chunk]),
                    #dest = i, tag = j)
            ## move to the next chunk of the array
            current_left_index += N_this_chunk

        ## update rank 0's copy of the lists to only its chunk
        nH_arr = copy.copy(nH_arr[:N_parts[0]])
        temperature_arr = copy.copy(temperature_arr[:N_parts[0]])
        metallicity_arr = copy.copy(metallicity_arr[:N_parts[0]])
        shieldLength_arr = copy.copy(shieldLength_arr[:N_parts[0]])
        init_chem_arr = copy.copy(init_chem_arr[:N_parts[0]])

        if driver_pars["UV_field"] == "StellarFluxes": 
            if driver_pars["compute_stellar_fluxes"] == 0: 
                ChimesFluxIon_arr= copy.copy(ChimesFluxIon_arr[:N_parts[0]]) 
                ChimesFluxG0_arr= copy.copy(ChimesFluxG0_arr[:N_parts[0]]) 
            else: 
                gas_coords_arr = copy.copy(gas_coords_arr[:N_parts[0]])
        elif driver_pars["UV_field"] == "S04" and driver_pars["IO_mode"] == "snapshot": 
            gas_coords_arr = copy.copy(gas_coords_arr[:N_parts[0]]) 

    #else:
        ## receive the correct chunk for each array
        #nH_arr = comm.recv(source = 0, tag=0)
        #temperature_arr = comm.recv(source = 0, tag=1)
        #metallicity_arr = comm.recv(source = 0, tag=2)
        #shieldLength_arr = comm.recv(source = 0, tag=3)
        #init_chem_arr = comm.recv(source = 0, tag=4)

        #if driver_pars["UV_field"] == "StellarFluxes": 
            #if driver_pars["compute_stellar_fluxes"] == 0:
                #ChimesFluxIon_arr = comm.recv(source = 0, tag = 5) 
                #ChimesFluxG0_arr = comm.recv(source = 0, tag = 6) 
            #else: 
                #gas_coords_arr = comm.recv(source = 0, tag = 5)
        #elif driver_pars["UV_field"] == "S04" and driver_pars["IO_mode"] == "snapshot": 
            #gas_coords_arr = comm.recv(source = 0, tag = 5) 

    ## make sure we received the correct number of particles
    for arr in [
        nH_arr,temperature_arr,metallicity_arr,
        shieldLength_arr,init_chem_arr]:
        assert len(arr) == N_parts[rank]

    if driver_pars["UV_field"] == "StellarFluxes": 
        if driver_pars["compute_stellar_fluxes"] == 0: 
            for arr in [ChimesFluxIon_arr, 
                        ChimesFluxG0_arr]: 
                assert len(arr) == N_parts[rank] 
        else: 
            assert len(gas_coords_arr) == N_parts[rank]
    elif driver_pars["UV_field"] == "S04" and driver_pars["IO_mode"] == "snapshot": 
        assert len(gas_coords_arr) == N_parts[rank] 

    #comm.Barrier()

    #if driver_pars["UV_field"] == "StellarFluxes" and driver_pars["compute_stellar_fluxes"] == 1 and N_star > 0: 
        # Broadcast all star particles to all MPI tasks 
        #star_coords_arr = comm.bcast(star_coords_arr, root = 0) 
        #star_mass_arr = comm.bcast(star_mass_arr, root = 0) 
        #star_age_Myr_arr = comm.bcast(star_age_Myr_arr, root = 0) 

        #comm.Barrier()

        if rank == 0: 
            print("Computing stellar fluxes")
            sys.stdout.flush()

        ChimesFluxIon_arr, ChimesFluxG0_arr = compute_stellar_fluxes(gas_coords_arr, star_coords_arr, star_mass_arr / solar_mass_cgs, star_age_Myr_arr, driver_pars["stellar_fluxes_fEsc_ion"], driver_pars["stellar_fluxes_fEsc_G0"], rank) 

        #comm.Barrier() 

        if rank == 0: 
            print("Finished computing stellar fluxes")
            sys.stdout.flush()

    ##  create and run an instance of ChimesDriver on different MPI tasks
    ##  each task uses its own chunk of the data
    data_list_thisTask = [nH_arr, temperature_arr, metallicity_arr, shieldLength_arr, init_chem_arr] 
    if driver_pars["UV_field"] == "StellarFluxes": 
        data_list_thisTask.append(ChimesFluxIon_arr) 
        data_list_thisTask.append(ChimesFluxG0_arr)
    elif driver_pars["UV_field"] == "S04" and driver_pars["IO_mode"] == "snapshot":
        distance_to_AGN = np.sqrt(((gas_coords_arr[:, 0] - driver_pars["AGN_position_x_kpc"]) ** 2.0) +
                                  ((gas_coords_arr[:, 1] - driver_pars["AGN_position_y_kpc"]) ** 2.0) +
                                  ((gas_coords_arr[:, 2] - driver_pars["AGN_position_z_kpc"]) ** 2.0))

        data_list_thisTask.append(distance_to_AGN) 

    my_driver = ChimesDriver(data_list_thisTask, driver_pars, global_variable_pars, 
                             gas_variable_pars, rank) 

    ## Check number of species in the network matches the number of  
    ## species in init_chem_arr 
    N_species = my_driver.myGlobalVars.totalNumberOfSpecies 
    
    if N_species == len(init_chem_arr[0]): 
        if rank == 0: 
            print("%d species in the network." % (N_species, ))
            sys.stdout.flush()
    else: 
        error_line = "ERROR: %d species in the network, but %d species in the initial abundance array. Aborting" % (N_species, len(init_chem_arr[0]))
        raise Exception(error_line) 

    final_output_array_thisTask, chimes_cumulative_time = my_driver.run()
    
    #comm.Barrier()

    ## Send the final output arrays from each MPI task to the 
    ## root task. A simple comm.gather() will fail if these 
    ## arrays exceed 2 GB, so it is easier to manually Send/Recv 
    ## from each task. 

    ## First, flatten the array to 1D. 
    final_output_array_thisTask_buffer = final_output_array_thisTask.flatten(order = 'C') 

    if rank == 0: 

        ## The buffer list will store buffers to 
        ## receive the arrays, one from each task. 
        final_output_array_buffer_list = [] 
        for i in range(N_task): 
            if driver_pars["driver_mode"] == "eqm_state" or driver_pars["driver_mode"] == "eqm_table": 
                final_output_array_buffer = np.empty(N_parts[i] * len(final_output_array_thisTask[0]), dtype = np.float64)
                shape = (N_parts[i], len(final_output_array_thisTask[0]))
            elif driver_pars["driver_mode"] == "cooling_rates": 
                final_output_array_buffer = np.empty(N_parts[i] * 2, dtype = np.float64) 
                shape = (N_parts[i], 2)
            elif driver_pars["driver_mode"] == "noneq_evolution": 
                final_output_array_buffer = np.empty(N_parts[i] * len(final_output_array_thisTask[0]) * len(final_output_array_thisTask[0][0]), dtype = np.float64) 
                shape = (N_parts[i], len(final_output_array_thisTask[0]), len(final_output_array_thisTask[0][0])) 
            else: 
                raise Exception("ERROR: driver_mode %s not recognised. Aborting" % (driver_pars["driver_mode"], )) 

            ## We need to store both the 1D buffer, and the 
            ## shape that will be needed to re-shape it back 
            ## into the multi-dimensional array at the end. 
            if i == 0: 
                final_output_array_buffer_list.append((final_output_array_thisTask_buffer, shape))  
            else: 
                final_output_array_buffer_list.append((final_output_array_buffer, shape)) 

        for i in range(1, N_task): 
            thisTag = i * 10 
            #comm.Recv([final_output_array_buffer_list[i][0], MPI.DOUBLE], source = i, tag = thisTag) 
    else: 
        thisTag = rank * 10 
        #comm.Send([final_output_array_thisTask_buffer, MPI.DOUBLE], dest = 0, tag = thisTag) 

    #comm.Barrier() 

    if rank == 0: 
        ## Rank 0 now needs to go through the 
        ## received 1D buffers and re-shape them. 
        final_output_array_list = [] 
        for i in range(N_task): 
            final_output_array_list.append(final_output_array_buffer_list[i][0].reshape(final_output_array_buffer_list[i][1], order = 'C')) 

        ## Finally, concatenate the individual 
        ## arrays from each task. 
        final_output_array = final_output_array_list[0] 
        if N_task > 1: 
            for i in range(1, N_task): 
                final_output_array = np.concatenate((final_output_array, final_output_array_list[i]), axis = 0) 
                
    ## Gather the elapsed time spent 
    ## by each task. 
    #chimes_total_time = comm.gather(chimes_cumulative_time, root = 0) 

    if driver_pars["UV_field"] == "StellarFluxes" and driver_pars["compute_stellar_fluxes"] == 1: 
        # Collect stellar fluxes on the root node 
        if rank == 0: 
            ion_recv_buffer = [] 
            G0_recv_buffer = [] 
            for i in range(1, N_task): 
                ion_recv_buffer.append(np.empty((N_parts[i], 8), dtype = np.float32)) 
                G0_recv_buffer.append(np.empty((N_parts[i], 8), dtype = np.float32)) 

       # if rank != 0: 
            #comm.send(ChimesFluxIon_arr, dest = 0, tag = 0) 
            #comm.send(ChimesFluxG0_arr, dest = 0, tag = 1) 
        #else: 
            #for i in range(1, N_task): 
                #ion_recv_buffer[i - 1] = comm.recv(source = i, tag = 0) 
                #G0_recv_buffer[i - 1] = comm.recv(source = i, tag = 1) 

        #comm.Barrier() 

        if rank == 0: 
            ChimesFluxIon_ALL = ChimesFluxIon_arr.copy() 
            ChimesFluxG0_ALL = ChimesFluxG0_arr.copy() 

            for i in range(1, N_task): 
                ChimesFluxIon_ALL = np.concatenate((ChimesFluxIon_ALL, ion_recv_buffer[i - 1])) 
                ChimesFluxG0_ALL = np.concatenate((ChimesFluxG0_ALL, G0_recv_buffer[i - 1])) 

    ## Write outputs 
    if rank == 0:
        #print("Total time spent in chemistry solver across all MPI ranks: %.4f seconds" % (sum(chimes_total_time, )))
        sys.stdout.flush()
        
        if driver_pars["IO_mode"] == "snapshot": 
            if driver_pars["driver_mode"] == "eqm_state": 
                # Write out eqm abundances to HDF5 file 
                with h5py.File(driver_pars["output_file"], 'a') as h5file_out:
                    array_name = "%s/EqmChemistryAbundances" % (driver_pars["hdf5_output_group"], ) 
                    h5file_out[array_name] = final_output_array

            elif driver_pars["driver_mode"] == "cooling_rates": 
                log_cool = np.log10(max(final_output_array[:, 0], 1.0e-100)) 
                log_heat = np.log10(max(final_output_array[:, 1], 1.0e-100)) 
                        
                with h5py.File(driver_pars["output_file"], 'a') as h5file_out: 
                    cool_array_name = "%s/log_cooling_rate" % (driver_pars["hdf5_output_group"], ) 
                    heat_array_name = "%s/log_heating_rate" % (driver_pars["hdf5_output_group"], ) 

                    h5file_out["log_cooling_rate"] = np.float32(log_cool)
                    h5file_out["log_heating_rate"] = np.float32(log_heat)
                    
            elif driver_pars["driver_mode"] == "noneq_evolution": 
                abundance_array = final_output_array[:, :N_species, :] 
                temperature_array = final_output_array[:, N_species, :] 
                time_array = np.array([i * gas_variable_pars["hydro_timestep"] for i in range(driver_pars["n_iterations"] + 1)]) 

                with h5py.File(driver_pars["output_file"], 'a') as h5file_out: 
                    abundance_array_name = "%s/AbundanceEvolution" % (driver_pars["hdf5_output_group"], ) 
                    temperature_array_name = "%s/TemperatureEvolution" % (driver_pars["hdf5_output_group"], ) 
                    time_array_name = "%s/TimeArray_seconds" % (driver_pars["hdf5_output_group"], ) 

                    h5file_out[abundance_array_name] = abundance_array 
                    h5file_out[temperature_array_name] = temperature_array 
                    h5file_out[time_array_name] = time_array 

            else: 
                raise Exception("ERROR: IO_mode %s cannot be run with driver_mode %s. Aborting." % (driver_pars["IO_mode"], driver_pars["driver_mode"])) 

            if driver_pars["UV_field"] == "StellarFluxes" and driver_pars["compute_stellar_fluxes"] == 1: 
                with h5py.File(driver_pars["output_file"], 'a') as h5file_out:
                    if driver_pars["snapshot_flux_ion_array"] != None: 
                        h5file_out[driver_pars["snapshot_flux_ion_array"]] = ChimesFluxIon_ALL 

                    if driver_pars["snapshot_flux_G0_array"] != None: 
                        h5file_out[driver_pars["snapshot_flux_G0_array"]] = ChimesFluxG0_ALL 
            
        elif driver_pars["IO_mode"] == "grid": 
            # We will need to re-construct the 3D 
            # grid of (T, nH, Z) for the output 
            # arrays. 
            log_T = np.arange(driver_pars["log_T_min"], driver_pars["log_T_max"] + (driver_pars["delta_log_T"] / 10.0), driver_pars["delta_log_T"]) 
            log_nH = np.arange(driver_pars["log_nH_min"], driver_pars["log_nH_max"] + (driver_pars["delta_log_nH"] / 10.0), driver_pars["delta_log_nH"]) 
            log_Z = np.arange(driver_pars["log_Z_min"], driver_pars["log_Z_max"] + (driver_pars["delta_log_Z"] / 10.0), driver_pars["delta_log_Z"]) 
            
            dim_T = len(log_T) 
            dim_nH = len(log_nH) 
            dim_Z = len(log_Z) 

            if driver_pars["driver_mode"] == "eqm_table" or driver_pars["driver_mode"] == "eqm_state": 
                abundance_array = np.zeros((dim_T, dim_nH, dim_Z, N_species), dtype = np.float64) 
            
                for i in range(dim_T): 
                    for j in range(dim_nH): 
                        for k in range(dim_Z): 
                            array_index = (i * dim_nH * dim_Z) + (j * dim_Z) + k 
                            for abun_index in range(N_species): 
                                abundance_array[i, j, k, abun_index] = max(final_output_array[array_index, abun_index], 1.0e-38)  
                        
                with h5py.File(driver_pars["output_file"], 'w') as h5file_out: 
                    if driver_pars["driver_mode"] == "eqm_table": 
                        # Store log Abundances
                        h5file_out["Abundances"] = np.float32(np.log10(abundance_array)) 
                    else: 
                        # Store linear Abundances
                        h5file_out["Abundances"] = abundance_array 

                    h5file_out["TableBins/Temperatures"] = np.float32(log_T) 
                    h5file_out["TableBins/Densities"] = np.float32(log_nH) 
                    h5file_out["TableBins/Metallicities"] = np.float32(log_Z) 
                    h5file_out["TableBins/N_Temperatures"] = dim_T 
                    h5file_out["TableBins/N_Densities"] = dim_nH 
                    h5file_out["TableBins/N_Metallicities"] = dim_Z 
                    h5file_out["TableBins/N_species"] = N_species

                    # Create Header information 
                    header_grp = h5file_out.create_group("Header")

                    header_grp.attrs["version_date"] = driver_pars["version_date"] 
                    header_grp.attrs["chimes_webpage"] = "https://richings.bitbucket.io/chimes/home.html"
                    
                    if driver_pars["UV_field"] == "Colibre" or driver_pars["shield_mode"] == "Colibre" or driver_pars["dust_depletion"] == "Colibre": 
                        header_grp.attrs["colibre_N_min"] = 3.08567758e15
                        header_grp.attrs["colibre_N_max"] = 1.0e24
                        header_grp.attrs["colibre_log_T_min"] = driver_pars["colibre_log_T_min"]
                        header_grp.attrs["colibre_log_T_max"] = driver_pars["colibre_log_T_max"]
                    
            elif driver_pars["driver_mode"] == "cooling_rates": 
                log_cool = np.zeros((dim_T, dim_nH, dim_Z)) 
                log_heat = np.zeros((dim_T, dim_nH, dim_Z)) 
            
                for i in range(dim_T): 
                    for j in range(dim_nH): 
                        for k in range(dim_Z): 
                            array_index = (i * dim_nH * dim_Z) + (j * dim_Z) + k 
                            log_cool[i, j, k] = np.log10(max(final_output_array[array_index, 0], 1.0e-100)) 
                            log_heat[i, j, k] = np.log10(max(final_output_array[array_index, 1], 1.0e-100)) 
                        
                with h5py.File(driver_pars["output_file"], 'w') as h5file_out: 
                    h5file_out["log_cooling_rate"] = np.float32(log_cool)
                    h5file_out["log_heating_rate"] = np.float32(log_heat)
                    h5file_out["TableBins/Temperatures"] = np.float32(log_T) 
                    h5file_out["TableBins/Densities"] = np.float32(log_nH) 
                    h5file_out["TableBins/Metallicities"] = np.float32(log_Z) 
                    h5file_out["TableBins/N_Temperatures"] = dim_T 
                    h5file_out["TableBins/N_Densities"] = dim_nH 
                    h5file_out["TableBins/N_Metallicities"] = dim_Z 

            elif driver_pars["driver_mode"] == "noneq_evolution": 
            
                abundance_array = np.zeros((dim_T, dim_nH, dim_Z, N_species, driver_pars["n_iterations"] + 1), dtype = np.float64) 
                temperature_array = np.zeros((dim_T, dim_nH, dim_Z, driver_pars["n_iterations"] + 1), dtype = np.float64) 
            
                for i in range(dim_T): 
                    for j in range(dim_nH): 
                        for k in range(dim_Z): 
                            array_index = (i * dim_nH * dim_Z) + (j * dim_Z) + k 
                            for t in range(driver_pars["n_iterations"] + 1): 
                                for abun_index in range(N_species): 
                                    abundance_array[i, j, k, abun_index, t] = final_output_array[array_index, abun_index, t] 
                                    temperature_array[i, j, k, t] = final_output_array[array_index, N_species, t] 

            
                time_array = np.array([i * gas_variable_pars["hydro_timestep"] for i in range(driver_pars["n_iterations"] + 1)]) 
                
                with h5py.File(driver_pars["output_file"], 'w') as h5file_out: 
                    h5file_out["AbundanceEvolution"] = abundance_array 
                    h5file_out["TemperatureEvolution"] = temperature_array 
                    h5file_out["TimeArray_seconds"] = time_array 
                    h5file_out["TableBins/Temperatures"] = np.float32(log_T) 
                    h5file_out["TableBins/Densities"] = np.float32(log_nH) 
                    h5file_out["TableBins/Metallicities"] = np.float32(log_Z) 
                    h5file_out["TableBins/N_Temperatures"] = dim_T 
                    h5file_out["TableBins/N_Densities"] = dim_nH 
                    h5file_out["TableBins/N_Metallicities"] = dim_Z 

            else: 
                raise Exception("ERROR: IO_mode %s cannot be run with driver_mode %s. Aborting." % (driver_pars["IO_mode"], driver_pars["driver_mode"])) 

            print("Finished!")
            sys.stdout.flush()

    return 
    
if __name__ == '__main__':
    main()
