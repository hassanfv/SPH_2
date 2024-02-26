import numpy as np 

##### Parameters that control the behaviour of chimes-plotter ####
plotter_parameters = {
    "plotter_mode" : "eqm_table",  # Options: eqm_table, cooling_rates, noneq_evolution_temperature, noneq_evolution_species 
    "input_file" : None,
    "output_file" : None,
    "species_list" : [],
    "x_variable" : "temperature",        # Options: temperature, density, metallicity
    "log_x_min" : 2.0, 
    "log_x_max" : 4.0, 
    "log_y_min" : -8.0, 
    "log_y_max" : 0.5, 
    "panel_variable" : "density",        # Options: temperature, density, metallicity
    "N_panels" : 6,
    "panel_var_min" : -2.0,
    "panel_var_max" : 2.0,
    "line_variable" : "metallicity",     # Options: temperature, density, metallicity
    "N_lines" : 2,
    "line_var_min" : -0.5,
    "line_var_max" : 0.0, 
    "fixed_variable" : None,             # Options: temperature, density, metallicity
    "fixed_var_value" : 0.0, 
    "N_T_init" : 2, 
    "log_T_init_min" : 2.0, 
    "log_T_init_max" : 4.0,

    # Flags to control which elements
    # are included in the network. 
    "element_included" : (1, 1, 1, 1, 1, 1, 1, 1, 1), 

    # The following parameters control the 
    # appearance of the figures. They can 
    # generally be left to their default 
    # values, but the user can edit them 
    # via the parameter file if required. 
    "figure_width" : 8.0,
    "figure_height" : 5.0,
    "subplot_left" : 0.08, 
    "subplot_right" : 0.8, 
    "subplot_bottom" : 0.1, 
    "subplot_top" : 0.95, 
    "line_width" : 1.8,  
    "tick_label_fontsize" : 14, 
    "axis_label_fontsize" : 14, 
    "panel_label_fontsize" : 9, 
    "legend_fontsize" : 9, 
    "panel_label_x" : 0.98, 
    "panel_label_y" : 0.02, 
    "max_ticks" : 6 
}

def read_parameters(infile):
    fd = open(infile, "r")

    element_included = np.ones(9, dtype = np.int32)
    
    for line in fd:
        if len(line.strip()) < 1:
            continue
        elif line.strip()[0] == "#":
            continue
        else:
            values = line.split()

            if values[0] in plotter_parameters:
                # Try to evaluate the parameter as a number,
                # otherwise keep it as a string.
                try:
                    plotter_parameters[values[0]] = eval(values[1])
                except:
                    plotter_parameters[values[0]] = values[1]

            elif values[0] == "species": 
                # The species parameter can appear
                # multiple times. These need to
                # be combined into a single list.
                plotter_parameters["species_list"].append(values[1])
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
                raise KeyError("Parameter %s not recognised. Aborting!" % (values[0], ))

    element_included_tuple = tuple([i for i in element_included])
    plotter_parameters["element_included"] = element_included_tuple 
        
    fd.close()

    return plotter_parameters

def print_parameters(infile, plotter_pars):
    print("####################") 
    print("#### Parameters ####") 
    print("####################") 

    print("Parameter file: %s" % (infile, )) 
    print(" ")

    print("Plotter parameters:")
    for key in plotter_pars:
        print(key, plotter_pars[key])

    print(" ")

    return
