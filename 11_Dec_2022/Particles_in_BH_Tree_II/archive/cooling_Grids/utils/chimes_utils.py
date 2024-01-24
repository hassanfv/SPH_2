import numpy as np
import ctypes as ct

from chimes_classes import *

def set_initial_chemistry_abundances(Z_array, global_pars, init_ion_state): 
    # Set the initial chemical abundances according to the 
    # init_ion_state parameter as follows: 
    # 0 - fully neutral and atomic. 
    # 1 - singly ionised. 
    # 2 - double ionised (where possible, otherwise singly ionised). 
    # etc. 

    # We use the Include... flags to determine the total number of 
    # species in the network (which may be less than 157 if some of 
    # the elements have been turned off). However, the Z_array still 
    # needs to contain all of the elements. 
    N_species = 10    # H + He always included. 
    if global_pars["element_included"][0] == 1: 
        N_species += 14  # C
    if global_pars["element_included"][1] == 1: 
        N_species += 8   # N 
    if global_pars["element_included"][2] == 1: 
        N_species += 17  # O
    if global_pars["element_included"][3] == 1: 
        N_species += 11  # Ne 
    if global_pars["element_included"][4] == 1: 
        N_species += 13  # Mg 
    if global_pars["element_included"][5] == 1: 
        N_species += 15  # Si 
    if global_pars["element_included"][6] == 1: 
        N_species += 17  # S 
    if global_pars["element_included"][7] == 1: 
        N_species += 21  # Ca 
    if global_pars["element_included"][8] == 1: 
        N_species += 27  # Fe 

    if global_pars["element_included"][0] == 1 and global_pars["element_included"][2] == 1: 
        N_species += 4 

    # Determine the metal number abundances relative to hydrogen 
    # (i.e. n_i / n_H) from the metallicity array (which is in 
    # terms of mass fractions relative to the total mass). 

    XH = 1.0 - (Z_array[:, 0] + Z_array[:, 1]) 
    N_part = len(XH) 
    atomic_masses = np.array([4., 12., 14., 16., 20., 24., 28., 32., 40., 56.])

    metal_number_abundances = np.zeros((N_part, 10), dtype = np.float32) 
    for i in range(N_part): 
        metal_number_abundances[i, :] = Z_array[i, 1:11] / (atomic_masses * XH[i])

    init_abundances = np.zeros((N_part, N_species), dtype = np.float32) 
    electron_abundance = np.zeros(N_part, dtype = np.float32) 

    init_abundances[:, 1 + min(init_ion_state, 1)] = 1.0   # H 
    electron_abundance += min(init_ion_state, 1.0) 

    init_abundances[:, 4 + min(init_ion_state, 2)] = metal_number_abundances[:, 0]   # He 
    electron_abundance += min(init_ion_state, 2) * metal_number_abundances[:, 0] 

    # current_index needs to account for which elements are included 
    current_index = 7 
    if global_pars["element_included"][0] == 1: 
        init_abundances[:, current_index + min(init_ion_state, 6)] = metal_number_abundances[:, 1]   # C 
        electron_abundance += min(init_ion_state, 6) * metal_number_abundances[:, 1] 
        current_index += 8 

    if global_pars["element_included"][1] == 1: 
        init_abundances[:, current_index + min(init_ion_state, 7)] = metal_number_abundances[:, 2]  # N 
        electron_abundance += min(init_ion_state, 7) * metal_number_abundances[:, 2] 
        current_index += 8 

    if global_pars["element_included"][2] == 1: 
        init_abundances[:, current_index + min(init_ion_state, 8)] = metal_number_abundances[:, 3]  # O 
        electron_abundance += min(init_ion_state, 8) * metal_number_abundances[:, 3] 
        current_index += 10 

    if global_pars["element_included"][3] == 1: 
        init_abundances[:, current_index + min(init_ion_state, 10)] = metal_number_abundances[:, 4]  # Ne 
        electron_abundance += min(init_ion_state, 10) * metal_number_abundances[:, 4] 
        current_index += 11 

    if global_pars["element_included"][4] == 1: 
        init_abundances[:, current_index + min(init_ion_state, 12)] = metal_number_abundances[:, 5]  # Mg 
        electron_abundance += min(init_ion_state, 12) * metal_number_abundances[:, 5] 
        current_index += 13 

    if global_pars["element_included"][5] == 1: 
        init_abundances[:, current_index + min(init_ion_state, 14)] = metal_number_abundances[:, 6]  # Si 
        electron_abundance += min(init_ion_state, 14) * metal_number_abundances[:, 6] 
        current_index += 15 

    if global_pars["element_included"][6] == 1: 
        init_abundances[:, current_index + min(init_ion_state, 16)] = metal_number_abundances[:, 7]  # S 
        electron_abundance += min(init_ion_state, 16) * metal_number_abundances[:, 7] 
        current_index += 17 

    if global_pars["element_included"][7] == 1: 
        init_abundances[:, current_index + min(init_ion_state, 20)] = metal_number_abundances[:, 8]  # Ca 
        electron_abundance += min(init_ion_state, 20) * metal_number_abundances[:, 8] 
        current_index += 21 

    if global_pars["element_included"][8] == 1: 
        init_abundances[:, current_index + min(init_ion_state, 26)] = metal_number_abundances[:, 9] # Fe 
        electron_abundance += min(init_ion_state, 26) * metal_number_abundances[:, 9] 

    init_abundances[:, 0] = electron_abundance 

    return init_abundances 

def compute_cooling_rates(myGasVars, myGlobalVars, chimesLib): 
    # Uses CHIMES routines to compute cooling and heating rates 
    chimesLib.allocate_current_rates_memory.argtypes = [ct.POINTER(chimes_current_rates_struct), 
                                                        ct.POINTER(globalVariables)] 

    chimesLib.free_current_rates_memory.argtypes = [ct.POINTER(chimes_current_rates_struct), 
                                                    ct.POINTER(globalVariables)] 
    
    chimesLib.set_initial_rate_coefficients.argtypes = [ct.POINTER(gasVariables), 
                                                       ct.POINTER(globalVariables), 
                                                       UserData] 

    chimesLib.update_rates.argtypes = [ct.POINTER(gasVariables), 
                                       ct.POINTER(globalVariables), 
                                       UserData] 

    chimesLib.calculate_total_cooling_rate.argtypes = [ct.POINTER(gasVariables), 
                                                       ct.POINTER(globalVariables), 
                                                       UserData, 
                                                       ct.c_int]     
    chimesLib.calculate_total_cooling_rate.restype = ChimesFloat 


    # Create instances of structures that we will need 
    myData = UserData() 
    mySpecies = Species_Structure() 
    myCurrentRates = chimes_current_rates_struct() 

    chimesLib.allocate_current_rates_memory(myCurrentRates, myGlobalVars)
    
    # Set up myData     
    myData.myGasVars = ct.pointer(myGasVars)
    myData.myGlobalVars = ct.pointer(myGlobalVars) 
    myData.species = ct.pointer(mySpecies) 
    myData.chimes_current_rates = ct.pointer(myCurrentRates) 
    
    if myGasVars.temperature <= myGlobalVars.Tmol_K: 
        myData.mol_flag_index = 1 
    else: 
        myData.mol_flag_index = 0 
    
    # Column densities 
    if myGlobalVars.cellSelfShieldingOn > 0: 
        NHtot = myGasVars.nH_tot * myGasVars.cell_size 
    else: 
        NHtot = 0.0 

    myData.HI_column = NHtot * myGasVars.abundances[myGlobalVars.speciesIndices[1]]
    myData.HeI_column = NHtot * myGasVars.abundances[myGlobalVars.speciesIndices[4]]
    myData.HeII_column = NHtot * myGasVars.abundances[myGlobalVars.speciesIndices[5]]
    myData.H2_column = NHtot * myGasVars.abundances[myGlobalVars.speciesIndices[137]]

    if myGlobalVars.speciesIndices[148] >= 0: 
        myData.CO_column = NHtot * myGasVars.abundances[myGlobalVars.speciesIndices[148]]
    else: 
        myData.CO_column = 0.0 

    if myGlobalVars.speciesIndices[141] >= 0: 
        myData.H2O_column = NHtot * myGasVars.abundances[myGlobalVars.speciesIndices[141]]
    else: 
        myData.H2O_column = 0.0 

    if myGlobalVars.speciesIndices[140] >= 0: 
        myData.OH_column = NHtot * myGasVars.abundances[myGlobalVars.speciesIndices[140]]
    else: 
        myData.OH_column = 0.0 

    myData.extinction = 4.0e-22 * NHtot * myGasVars.metallicity

    # Determine case A or B 
    if 6.3463e-18 * myData.HI_column < 1.0: 
        myData.case_AB_index[0] = 0 
    else: 
        myData.case_AB_index[0] = 1

    if 7.4347e-18 * myData.HeI_column < 1.0: 
        myData.case_AB_index[1] = 0 
    else: 
        myData.case_AB_index[1] = 1

    # The ThermEvolOn flag needs to be set to 
    # 1 to compute photoheating rates. 
    ThermEvolOn_save = myGasVars.ThermEvolOn
    myGasVars.ThermEvolOn = 1 

    chimesLib.set_initial_rate_coefficients(myData.myGasVars, myData.myGlobalVars, myData) 
    chimesLib.update_rates(myData.myGasVars, myData.myGlobalVars, myData) 

    # Calculate cooling and heating rates 
    cool_rate = chimesLib.calculate_total_cooling_rate(myData.myGasVars, myData.myGlobalVars, myData, 1)
    heat_rate = chimesLib.calculate_total_cooling_rate(myData.myGasVars, myData.myGlobalVars, myData, 2)

    # Reset ThermEvolOn to its 
    # original value. 
    myGasVars.ThermEvolOn = ThermEvolOn_save 

    chimesLib.free_current_rates_memory(myCurrentRates, myGlobalVars) 

    return cool_rate, heat_rate 
