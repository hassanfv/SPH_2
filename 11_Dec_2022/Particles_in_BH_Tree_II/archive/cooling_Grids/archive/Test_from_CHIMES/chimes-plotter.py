## Import from standard libraries 
import numpy as np
import sys

## import parameter routines
from plotter_config import read_parameters, print_parameters

## import eqm table routines
from eqm_table_routines import plot_eqm_table 

## import cooling rates routines 
from cooling_rates_routines import plot_cooling_rates 

## import noneq evolution routines 
from noneq_evolution_routines import plot_temperature_evolution, plot_species_evolution 

def main():
    ## Parse parameter file
    parameter_file = sys.argv[1]
    plotter_pars = read_parameters(parameter_file)

    print_parameters(parameter_file, plotter_pars)

    if plotter_pars["plotter_mode"] == "eqm_table":
        plot_eqm_table(plotter_pars)
    elif plotter_pars["plotter_mode"] == "cooling_rates": 
        plot_cooling_rates(plotter_pars) 
    elif plotter_pars["plotter_mode"] == "noneq_evolution_temperature": 
        plot_temperature_evolution(plotter_pars) 
    elif plotter_pars["plotter_mode"] == "noneq_evolution_species": 
        plot_species_evolution(plotter_pars) 
    else:
        raise KeyError("plotter_mode %s not recognised. Aborting." % (plotter_pars["plotter_mode"], ))

    print("Finished!") 

    return

if __name__ == "__main__":
    main()
    
