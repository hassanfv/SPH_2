import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import sys
import os
import h5py
import numpy as np 

try: 
    sys.path.insert(1, os.path.join(sys.path[0], '..')) 
    import read_chimes.read_chimes as rc
except ImportError: 
    sys.path.insert(1, os.path.join(sys.path[0], '..', 'read_chimes')) 
    import read_chimes as rc

from plotter_utils import decompose_2d_factors 

try: 
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['xtick.top'] = True
    matplotlib.rcParams['xtick.bottom'] = True
    matplotlib.rcParams['xtick.major.top'] = True
    matplotlib.rcParams['xtick.major.bottom'] = True
    matplotlib.rcParams['xtick.minor.top'] = True
    matplotlib.rcParams['xtick.minor.bottom'] = True
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['ytick.left'] = True
    matplotlib.rcParams['ytick.right'] = True
    matplotlib.rcParams['ytick.major.left'] = True
    matplotlib.rcParams['ytick.major.right'] = True
    matplotlib.rcParams['ytick.minor.left'] = True
    matplotlib.rcParams['ytick.minor.right'] = True
except KeyError: 
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'

convert_seconds_to_Myr = 1.0 / (60. * 60. * 24. * 365.25 * 1.0e6) 

def plot_temperature_evolution(plotter_pars): 
    ## Read in equilibrium table
    h5file = h5py.File(plotter_pars["input_file"], "r")

    array_dimensions = {"initial_temperature" : None,
                        "density" : None,
                        "metallicity" : None} 

    array_dimensions["initial_temperature"] = 10.0 ** np.array(h5file["TableBins/Temperatures"]) 
    array_dimensions["density"] = 10.0 ** np.array(h5file["TableBins/Densities"]) 
    array_dimensions["metallicity"] = 10.0 ** np.array(h5file["TableBins/Metallicities"])         
    
    # Read in time array 
    time_array = np.array(h5file["TimeArray_seconds"]) 
    T_evolution = np.array(h5file["TemperatureEvolution"]) 

    time_array *= convert_seconds_to_Myr 

    ## The dimensions of the T_evolution array 
    ## need to be re-ordered according to which 
    ## variable is plotted on with different lines, 
    ## and on different panels. initial_temperature 
    ## will always come first, and time last. 
    dimension_positions = {"initial_temperature" : 0,
                           "density" : 1,
                           "metallicity" : 2, 
                           "time" : 3}

    transpose_order = (dimension_positions["initial_temperature"],
                       dimension_positions[plotter_pars["line_variable"]],
                       dimension_positions[plotter_pars["panel_variable"]], 
                       dimension_positions["time"])

    T_evolution = np.transpose(h5file["TemperatureEvolution"], transpose_order) 
    
    h5file.close()

    # Determine line style indices
    line_styles = ["-", "--", ":", "-.", (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (5, 10)), (0, (1, 10)), (0, (3, 10, 1, 10)), (0, (3, 10, 1, 10, 1, 10))]
    if plotter_pars["N_lines"] > len(line_styles):
        raise Exception("ERROR: N_lines = %d, but there are only %d available line styles. Aborting." % (plotter_pars["N_lines"], len(line_styles)))

    line_var = np.log10(array_dimensions[plotter_pars["line_variable"]])
    line_index_min = np.abs(line_var - plotter_pars["line_var_min"]).argmin()
    line_index_max = np.abs(line_var - plotter_pars["line_var_max"]).argmin()

    line_var_indices = []
    if plotter_pars["N_lines"] > 1:
        for i in range(plotter_pars["N_lines"]):
            line_var_indices.append(line_index_min + ((i * (line_index_max - line_index_min)) // (plotter_pars["N_lines"] - 1)))
    else:
        line_var_indices.append(line_index_min) 

    ## Determine how to arrange 
    ## the panels of the figure. 
    subplot_grid = decompose_2d_factors(plotter_pars["N_panels"])
    subplot_grid_rows = min(subplot_grid)
    subplot_grid_cols = max(subplot_grid)

    panel_var = np.log10(array_dimensions[plotter_pars["panel_variable"]]) 
    panel_index_min = np.abs(panel_var - plotter_pars["panel_var_min"]).argmin()
    panel_index_max = np.abs(panel_var - plotter_pars["panel_var_max"]).argmin() 

    panel_var_indices = []
    if plotter_pars["N_panels"] > 1: 
        for i in range(plotter_pars["N_panels"]):
            panel_var_indices.append(panel_index_min + ((i * (panel_index_max - panel_index_min)) // (plotter_pars["N_panels"] - 1)))
    else: 
        panel_var_indices.append(panel_index_min)

    ## Determine which T_init curves to plot 
    T_init = np.log10(array_dimensions["initial_temperature"]) 
    T_init_index_min = np.abs(T_init - plotter_pars["log_T_init_min"]).argmin()
    T_init_index_max = np.abs(T_init - plotter_pars["log_T_init_max"]).argmin() 

    T_init_indices = [] 
    if plotter_pars["N_T_init"] > 1: 
        for i in range(plotter_pars["N_T_init"]): 
            T_init_indices.append(T_init_index_min + ((i * (T_init_index_max - T_init_index_min)) // (plotter_pars["N_T_init"] - 1))) 
    else: 
        T_init_indices.append(T_init_index_max) 

    
    ## Create figure
    fig, ax = plt.subplots(subplot_grid_rows, subplot_grid_cols, sharex = 'col', sharey = 'row', figsize = (plotter_pars["figure_width"], plotter_pars["figure_height"]))

    fig.subplots_adjust(hspace = 0, wspace = 0, left = plotter_pars["subplot_left"], right = plotter_pars["subplot_right"], bottom = plotter_pars["subplot_bottom"], top = plotter_pars["subplot_top"]) 
    
    panel_index = 0 
    for i in range(subplot_grid_rows):
        for j in range(subplot_grid_cols):
            for k in range(plotter_pars["N_T_init"]):
                for l in range(plotter_pars["N_lines"]): 
                    if plotter_pars["N_T_init"] > 1:
                        index_low = 10
                        index_hi = 245
                        col_index = index_low + ((k * (index_hi - index_low)) // (plotter_pars["N_T_init"] - 1))
                    else:
                        col_index = 128 

                    try: 
                        try:
                          
                            ax[i, j].plot((time_array[1:]), np.log10(T_evolution[T_init_indices[k], line_var_indices[l], panel_var_indices[panel_index], 1:]), linestyle = line_styles[l], color = matplotlib.cm.viridis(col_index), linewidth = plotter_pars["line_width"])
                        except IndexError: 
                            ax[j].plot((time_array[1:]), np.log10(T_evolution[T_init_indices[k], line_var_indices[l], panel_var_indices[panel_index], 1:]), linestyle = line_styles[l], color = matplotlib.cm.viridis(col_index), linewidth = plotter_pars["line_width"])
                    except TypeError: 
                        ax.plot((time_array[1:]), np.log10(T_evolution[T_init_indices[k], line_var_indices[l], panel_var_indices[panel_index], 1:]), linestyle = line_styles[l], color = matplotlib.cm.viridis(col_index), linewidth = plotter_pars["line_width"])

            # Set the panel label 
            panel_value = np.log10(array_dimensions[plotter_pars["panel_variable"]][panel_var_indices[panel_index]]) 
            if plotter_pars["panel_variable"] == "density": 
                panel_label = r"$\log_{10} n_{\rm{H}} = %.1f$" % (panel_value, ) 
            elif plotter_pars["panel_variable"] == "metallicity": 
                panel_label = r"$\log_{10} (Z / \rm{Z}_{\odot}) = %.1f$" % (panel_value, ) 
            else: 
                raise Exception("ERROR: panel_variable %s not recognised. Aborting" % (plotter_vars["panel_variable"], )) 

            try: 
                try: 
                    plt.text(plotter_pars["panel_label_x"], plotter_pars["panel_label_y"], panel_label, fontsize = plotter_pars["panel_label_fontsize"], horizontalalignment = 'right', verticalalignment = 'bottom', transform = ax[i, j].transAxes)
                except IndexError: 
                    plt.text(plotter_pars["panel_label_x"], plotter_pars["panel_label_y"], panel_label, fontsize = plotter_pars["panel_label_fontsize"], horizontalalignment = 'right', verticalalignment = 'bottom', transform = ax[j].transAxes)
            except TypeError: 
                plt.text(plotter_pars["panel_label_x"], plotter_pars["panel_label_y"], panel_label, fontsize = plotter_pars["panel_label_fontsize"], horizontalalignment = 'right', verticalalignment = 'bottom', transform = ax.transAxes) 

            panel_index += 1

    # Set axis ranges and tick labels 
    delta_x_tick = 1.0 
    x_ticks = [i for i in np.arange(round(plotter_pars["log_x_min"]), plotter_pars["log_x_max"] + (0.1 * delta_x_tick), delta_x_tick)] 
    while len(x_ticks) > plotter_pars["max_ticks"]: 
        delta_x_tick *= 2.0 
        x_ticks = [i for i in np.arange(round(plotter_pars["log_x_min"]), plotter_pars["log_x_max"] + (0.1 * delta_x_tick), delta_x_tick)] 

    x_ticks_minor = np.array(x_ticks)[:-1] + (delta_x_tick / 2.0)

    delta_y_tick = 1.0
    y_ticks = [i for i in np.arange(round(plotter_pars["log_y_min"]), plotter_pars["log_y_max"] + (0.1 * delta_y_tick), delta_y_tick)] 
    while len(y_ticks) > plotter_pars["max_ticks"]: 
        delta_y_tick *= 2.0 
        y_ticks = [i for i in np.arange(round(plotter_pars["log_y_min"]), plotter_pars["log_y_max"] + (0.1 * delta_y_tick), delta_y_tick)] 

    y_ticks_minor = np.array(y_ticks)[:-1] + (delta_y_tick / 2.0) 

    x_tick_labels = [] 
    x_tick_labels_reduced = [] 
    for i in range(len(x_ticks)): 
        x_tick_labels.append(r"$%.0f$" % (x_ticks[i], )) 
        if i == 0: 
            x_tick_labels_reduced.append(r"") 
        else: 
            x_tick_labels_reduced.append(x_tick_labels[-1]) 

    y_tick_labels = [] 
    y_tick_labels_reduced = [] 
    for i in range(len(y_ticks)): 
        y_tick_labels.append(r"$%.0f$" % (y_ticks[i], )) 
        if i == 0: 
            y_tick_labels_reduced.append(r"") 
        else: 
            y_tick_labels_reduced.append(y_tick_labels[-1]) 
        
    for i in range(subplot_grid_rows):
        try:
            try: 
                this_ax = ax[i, 0]
            except IndexError: 
                this_ax = ax[0]
        except TypeError:
            this_ax = ax 
            
        this_ax.set_ylim([plotter_pars["log_y_min"], plotter_pars["log_y_max"]]) 
        this_ax.yaxis.set_ticks(y_ticks) 
        this_ax.yaxis.set_ticks(y_ticks_minor, minor = True) 
        if i == subplot_grid_rows - 1: 
            this_ax.yaxis.set_ticklabels(y_tick_labels, fontsize = plotter_pars["tick_label_fontsize"]) 
        else: 
            this_ax.yaxis.set_ticklabels(y_tick_labels_reduced, fontsize = plotter_pars["tick_label_fontsize"]) 

    for j in range(subplot_grid_cols):
        try:
            try: 
                this_ax = ax[subplot_grid_rows - 1, j]
            except IndexError: 
                this_ax = ax[j]
        except TypeError:
            this_ax = ax 
            
        this_ax.set_xlim([plotter_pars["log_x_min"], plotter_pars["log_x_max"]]) 
        this_ax.xaxis.set_ticks(x_ticks) 
        this_ax.xaxis.set_ticks(x_ticks_minor, minor = True) 
        if j == 0: 
            this_ax.xaxis.set_ticklabels(x_tick_labels, fontsize = plotter_pars["tick_label_fontsize"]) 
        else: 
            this_ax.xaxis.set_ticklabels(x_tick_labels_reduced, fontsize = plotter_pars["tick_label_fontsize"]) 


    # Set axis labels. To position these correctly, 
    # we create dummy axes, which makes it easier to 
    # position them relative to the subplots. 
    ax_dummy_x = plt.axes([plotter_pars["subplot_left"], 0.0, plotter_pars["subplot_right"] - plotter_pars["subplot_left"], 0.02]) 
    ax_dummy_x.set_axis_off()
    x_label = r"$\rm{Time} \, (\rm{Myr})$" 
    plt.text(0.5, 2.5, x_label, fontsize = plotter_pars["axis_label_fontsize"], horizontalalignment = 'center', verticalalignment = 'top', transform = ax_dummy_x.transAxes) 

    ax_dummy_y = plt.axes([0.0, plotter_pars["subplot_bottom"], 0.02, plotter_pars["subplot_top"] - plotter_pars["subplot_bottom"]]) 
    ax_dummy_y.set_axis_off()
    y_label = r"$\log_{10} [T \, (\rm{K})]$" 
    plt.text(1.0, 0.5, y_label, fontsize = plotter_pars["axis_label_fontsize"], horizontalalignment = 'center', verticalalignment = 'center', rotation = 90, transform = ax_dummy_y.transAxes) 

    # Create legends. We again create dummy 
    # axes for these, to position them to 
    # the right of the plots. 
    ax_dummy_leg1 = plt.axes([plotter_pars["subplot_right"] + 0.02, (plotter_pars["subplot_top"] + plotter_pars["subplot_bottom"]) / 2.0, 0.98 - plotter_pars["subplot_right"], (plotter_pars["subplot_top"] - plotter_pars["subplot_bottom"]) / 2.0]) 
    ax_dummy_leg1.set_axis_off() 

    fontP = matplotlib.font_manager.FontProperties() 
    fontP.set_size(plotter_pars["legend_fontsize"])

    leg1_handles = [] 
    leg1_labels = [] 
    for i in range(plotter_pars["N_T_init"]): 
        if plotter_pars["N_T_init"] > 1:
            index_low = 10
            index_hi = 245
            col_index = index_low + ((i * (index_hi - index_low)) // (plotter_pars["N_T_init"] - 1))
        else:
            col_index = 128 
        leg1_handles.append(matplotlib.lines.Line2D([0], [0], color = matplotlib.cm.viridis(col_index), lw = plotter_pars["line_width"])) 

        leg1_labels.append(r"$\log_{10} T_{\rm{init}} = %.1f$" % (np.log10(array_dimensions["initial_temperature"][T_init_indices[i]]), )) 

    plt.legend(leg1_handles, leg1_labels, loc = 'upper left', bbox_to_anchor=(-0.15, 1.0), prop = fontP, handlelength = 2.5, handletextpad = 0.05, columnspacing = 0.1, ncol = 1, frameon = False) 

    ax_dummy_leg2 = plt.axes([plotter_pars["subplot_right"] + 0.02, plotter_pars["subplot_bottom"], 0.98 - plotter_pars["subplot_right"], (plotter_pars["subplot_top"] - plotter_pars["subplot_bottom"]) / 2.0]) 
    ax_dummy_leg2.set_axis_off()

    leg2_handles = [] 
    leg2_labels = [] 
    for i in range(plotter_pars["N_lines"]): 
        leg2_handles.append(matplotlib.lines.Line2D([0], [0], color = 'k', lw = plotter_pars["line_width"], linestyle = line_styles[i])) 

        line_value = np.log10(array_dimensions[plotter_pars["line_variable"]][line_var_indices[i]])
        if plotter_pars["line_variable"] == "density": 
            line_label = r"$\log_{10} n_{\rm{H}} = %.1f$" % (line_value, ) 
        elif plotter_pars["line_variable"] == "metallicity": 
            line_label = r"$\log_{10} (Z / \rm{Z}_{\odot}) = %.1f$" % (line_value, )

        leg2_labels.append(line_label) 

    plt.legend(leg2_handles, leg2_labels, loc = 'upper left', bbox_to_anchor=(-0.15, 1.0), prop = fontP, handlelength = 2.5, handletextpad = 0.05, columnspacing = 0.1, ncol = 1, frameon = False) 

    plt.savefig(plotter_pars["output_file"], dpi = 300)
    plt.close()

    return

def plot_species_evolution(plotter_pars): 
    ## Read in equilibrium table
    h5file = h5py.File(plotter_pars["input_file"], "r")

    array_dimensions = {"temperature" : None,
                        "density" : None,
                        "metallicity" : None} 

    array_dimensions["temperature"] = 10.0 ** np.array(h5file["TableBins/Temperatures"]) 
    array_dimensions["density"] = 10.0 ** np.array(h5file["TableBins/Densities"]) 
    array_dimensions["metallicity"] = 10.0 ** np.array(h5file["TableBins/Metallicities"])         
    
    # Read in time array 
    time_array = np.array(h5file["TimeArray_seconds"]) 
    time_array *= convert_seconds_to_Myr

    ## The dimensions of the species evolution arrays 
    ## need to be re-ordered according to which 
    ## variable is plotted with different lines, 
    ## on different panels, and is fixed. Time is last. 
    dimension_positions = {"temperature" : 0,
                           "density" : 1,
                           "metallicity" : 2, 
                           "time" : 3}

    transpose_order = (dimension_positions[plotter_pars["fixed_variable"]],
                       dimension_positions[plotter_pars["line_variable"]],
                       dimension_positions[plotter_pars["panel_variable"]], 
                       dimension_positions["time"])

    ## List storing the abundance evolution 
    ## of individual species as separate arrays
    abundance_array_list = []
    
    # Create species dictionary for
    # reduced network
    reduced_chimes_dict = rc.create_reduced_chimes_dictionary(plotter_pars["element_included"])
    
    for chimes_species in plotter_pars["species_list"]:
        try:
            chimes_index = reduced_chimes_dict[chimes_species]
        except KeyError:
            print("Error: species %s is not recognised in the CHIMES abundance array. Aborting." % (chimes_species, ))
            return 

        species_array = np.transpose(h5file["AbundanceEvolution"][:, :, :, chimes_index, :], transpose_order)

        # Protect against log(0)
        species_array[(species_array < 1.0e-100)] = 1.0e-100 

        abundance_array_list.append(species_array)
    
    h5file.close()

    # Determine line style indices
    line_styles = ["-", "--", ":", "-.", (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (5, 10)), (0, (1, 10)), (0, (3, 10, 1, 10)), (0, (3, 10, 1, 10, 1, 10))]
    if plotter_pars["N_lines"] > len(line_styles):
        raise Exception("ERROR: N_lines = %d, but there are only %d available line styles. Aborting." % (plotter_pars["N_lines"], len(line_styles)))

    line_var = np.log10(array_dimensions[plotter_pars["line_variable"]])
    line_index_min = np.abs(line_var - plotter_pars["line_var_min"]).argmin()
    line_index_max = np.abs(line_var - plotter_pars["line_var_max"]).argmin()

    line_var_indices = []
    if plotter_pars["N_lines"] > 1:
        for i in range(plotter_pars["N_lines"]):
            line_var_indices.append(line_index_min + ((i * (line_index_max - line_index_min)) // (plotter_pars["N_lines"] - 1)))
    else:
        line_var_indices.append(line_index_min) 

    ## Determine how to arrange 
    ## the panels of the figure. 
    subplot_grid = decompose_2d_factors(plotter_pars["N_panels"])
    subplot_grid_rows = min(subplot_grid)
    subplot_grid_cols = max(subplot_grid)

    panel_var = np.log10(array_dimensions[plotter_pars["panel_variable"]]) 
    panel_index_min = np.abs(panel_var - plotter_pars["panel_var_min"]).argmin()
    panel_index_max = np.abs(panel_var - plotter_pars["panel_var_max"]).argmin() 

    panel_var_indices = []
    if plotter_pars["N_panels"] > 1: 
        for i in range(plotter_pars["N_panels"]):
            panel_var_indices.append(panel_index_min + ((i * (panel_index_max - panel_index_min)) // (plotter_pars["N_panels"] - 1)))
    else: 
        panel_var_indices.append(panel_index_min)

    # Determine the index of the fixed variable 
    fixed_var = np.log10(array_dimensions[plotter_pars["fixed_variable"]]) 
    fixed_var_index = np.abs(fixed_var - plotter_pars["fixed_var_value"]).argmin() 
    
    ## Create figure
    fig, ax = plt.subplots(subplot_grid_rows, subplot_grid_cols, sharex = 'col', sharey = 'row', figsize = (plotter_pars["figure_width"], plotter_pars["figure_height"]))

    fig.subplots_adjust(hspace = 0, wspace = 0, left = plotter_pars["subplot_left"], right = plotter_pars["subplot_right"], bottom = plotter_pars["subplot_bottom"], top = plotter_pars["subplot_top"]) 
    
    N_species = len(plotter_pars["species_list"])
    panel_index = 0 
    for i in range(subplot_grid_rows):
        for j in range(subplot_grid_cols):
            for k in range(N_species):
                for l in range(plotter_pars["N_lines"]): 
                    if N_species > 1:
                        index_low = 10
                        index_hi = 245
                        col_index = index_low + ((k * (index_hi - index_low)) // (N_species - 1))
                    else:
                        col_index = 128 


                    try: 
                        try: 
                            ax[i, j].plot(np.log10(time_array[1:]), np.log10(abundance_array_list[k][fixed_var_index, line_var_indices[l], panel_var_indices[panel_index], 1:]), linestyle = line_styles[l], color = matplotlib.cm.viridis(col_index), linewidth = plotter_pars["line_width"])
                        except IndexError: 
                            ax[j].plot(np.log10(time_array[1:]), np.log10(abundance_array_list[k][fixed_var_index, line_var_indices[l], panel_var_indices[panel_index], 1:]), linestyle = line_styles[l], color = matplotlib.cm.viridis(col_index), linewidth = plotter_pars["line_width"])
                    except TypeError: 
                        ax[i, j].plot(np.log10(time_array[1:]), np.log10(abundance_array_list[k][fixed_var_index, line_var_indices[l], panel_var_indices[panel_index], 1:]), linestyle = line_styles[l], color = matplotlib.cm.viridis(col_index), linewidth = plotter_pars["line_width"])

            # Set the panel label 
            panel_value = np.log10(array_dimensions[plotter_pars["panel_variable"]][panel_var_indices[panel_index]]) 
            if plotter_pars["panel_variable"] == "density": 
                panel_label = r"$\log_{10} n_{\rm{H}} = %.1f$" % (panel_value, ) 
            elif plotter_pars["panel_variable"] == "metallicity": 
                panel_label = r"$\log_{10} (Z / \rm{Z}_{\odot}) = %.1f$" % (panel_value, ) 
            else: 
                raise Exception("ERROR: panel_variable %s not recognised. Aborting" % (plotter_vars["panel_variable"], )) 

            try: 
                try: 
                    plt.text(plotter_pars["panel_label_x"], plotter_pars["panel_label_y"], panel_label, fontsize = plotter_pars["panel_label_fontsize"], horizontalalignment = 'right', verticalalignment = 'bottom', transform = ax[i, j].transAxes)
                except IndexError: 
                    plt.text(plotter_pars["panel_label_x"], plotter_pars["panel_label_y"], panel_label, fontsize = plotter_pars["panel_label_fontsize"], horizontalalignment = 'right', verticalalignment = 'bottom', transform = ax[j].transAxes)
            except TypeError: 
                plt.text(plotter_pars["panel_label_x"], plotter_pars["panel_label_y"], panel_label, fontsize = plotter_pars["panel_label_fontsize"], horizontalalignment = 'right', verticalalignment = 'bottom', transform = ax.transAxes) 

            panel_index += 1

    # Set axis ranges and tick labels 
    delta_x_tick = 1.0 
    x_ticks = [i for i in np.arange(round(plotter_pars["log_x_min"]), plotter_pars["log_x_max"] + (0.1 * delta_x_tick), delta_x_tick)] 
    while len(x_ticks) > plotter_pars["max_ticks"]: 
        delta_x_tick *= 2.0 
        x_ticks = [i for i in np.arange(round(plotter_pars["log_x_min"]), plotter_pars["log_x_max"] + (0.1 * delta_x_tick), delta_x_tick)] 

    x_ticks_minor = np.array(x_ticks)[:-1] + (delta_x_tick / 2.0)

    delta_y_tick = 1.0 
    y_ticks = [i for i in np.arange(round(plotter_pars["log_y_min"]), plotter_pars["log_y_max"] + (0.1 * delta_y_tick), delta_y_tick)] 
    while len(y_ticks) > plotter_pars["max_ticks"]: 
        delta_y_tick *= 2.0 
        y_ticks = [i for i in np.arange(round(plotter_pars["log_y_min"]), plotter_pars["log_y_max"] + (0.1 * delta_y_tick), delta_y_tick)] 

    y_ticks_minor = np.array(y_ticks)[:-1] + (delta_y_tick / 2.0) 

    x_tick_labels = [] 
    x_tick_labels_reduced = [] 
    for i in range(len(x_ticks)): 
        x_tick_labels.append(r"$%.0f$" % (x_ticks[i], )) 
        if i == 0: 
            x_tick_labels_reduced.append(r"") 
        else: 
            x_tick_labels_reduced.append(x_tick_labels[-1]) 

    y_tick_labels = [] 
    y_tick_labels_reduced = [] 
    for i in range(len(y_ticks)): 
        y_tick_labels.append(r"$%.0f$" % (y_ticks[i], )) 
        if i == 0: 
            y_tick_labels_reduced.append(r"") 
        else: 
            y_tick_labels_reduced.append(y_tick_labels[-1]) 
        
    for i in range(subplot_grid_rows):
        try:
            try: 
                this_ax = ax[i, 0]
            except IndexError: 
                this_ax = ax[0]
        except TypeError:
            this_ax = ax 
        
        this_ax.set_ylim([plotter_pars["log_y_min"], plotter_pars["log_y_max"]]) 
        this_ax.yaxis.set_ticks(y_ticks) 
        this_ax.yaxis.set_ticks(y_ticks_minor, minor = True) 
        if i == subplot_grid_rows - 1: 
            this_ax.yaxis.set_ticklabels(y_tick_labels, fontsize = plotter_pars["tick_label_fontsize"]) 
        else: 
            this_ax.yaxis.set_ticklabels(y_tick_labels_reduced, fontsize = plotter_pars["tick_label_fontsize"]) 

    for j in range(subplot_grid_cols):
        try:
            try: 
                this_ax = ax[subplot_grid_rows - 1, j]
            except IndexError: 
                this_ax = ax[j]
        except TypeError:
            this_ax = ax 
            
        this_ax.set_xlim([plotter_pars["log_x_min"], plotter_pars["log_x_max"]]) 
        this_ax.xaxis.set_ticks(x_ticks) 
        this_ax.xaxis.set_ticks(x_ticks_minor, minor = True) 
        if j == 0: 
            this_ax.xaxis.set_ticklabels(x_tick_labels, fontsize = plotter_pars["tick_label_fontsize"]) 
        else: 
            this_ax.xaxis.set_ticklabels(x_tick_labels_reduced, fontsize = plotter_pars["tick_label_fontsize"]) 

    # Set axis labels. To position these correctly, 
    # we create dummy axes, which makes it easier to 
    # position them relative to the subplots. 
    ax_dummy_x = plt.axes([plotter_pars["subplot_left"], 0.0, plotter_pars["subplot_right"] - plotter_pars["subplot_left"], 0.02]) 
    ax_dummy_x.set_axis_off() 

    x_label = r"$\log_{10} [\rm{Time} \, (\rm{Myr})]$" 

    plt.text(0.5, 2.5, x_label, fontsize = plotter_pars["axis_label_fontsize"], horizontalalignment = 'center', verticalalignment = 'top', transform = ax_dummy_x.transAxes) 

    ax_dummy_y = plt.axes([0.0, plotter_pars["subplot_bottom"], 0.02, plotter_pars["subplot_top"] - plotter_pars["subplot_bottom"]]) 
    ax_dummy_y.set_axis_off() 
    y_label = r"$\log_{10} [n_{\rm{i}} / n_{\rm{H,} \, \rm{tot}}]$" 
    plt.text(1.0, 0.5, y_label, fontsize = plotter_pars["axis_label_fontsize"], horizontalalignment = 'center', verticalalignment = 'center', rotation = 90, transform = ax_dummy_y.transAxes) 

    # Create legends. We again create dummy 
    # axes for these, to position them to 
    # the right of the plots. 
    ax_dummy_leg1 = plt.axes([plotter_pars["subplot_right"] + 0.02, (plotter_pars["subplot_top"] + plotter_pars["subplot_bottom"]) / 2.0, 0.98 - plotter_pars["subplot_right"], (plotter_pars["subplot_top"] - plotter_pars["subplot_bottom"] - 0.02) / 2.0]) 
    ax_dummy_leg1.set_axis_off() 

    fontP = matplotlib.font_manager.FontProperties() 
    fontP.set_size(plotter_pars["legend_fontsize"])

    leg1_handles = [] 
    leg1_labels = [] 
    for i in range(N_species): 
        if N_species > 1:
            index_low = 10
            index_hi = 245
            col_index = index_low + ((i * (index_hi - index_low)) // (N_species - 1))
        else:
            col_index = 128 
        leg1_handles.append(matplotlib.lines.Line2D([0], [0], color = matplotlib.cm.viridis(col_index), lw = plotter_pars["line_width"])) 

        leg1_labels.append(r"$\rm{%s}$" % (plotter_pars["species_list"][i], )) 

    plt.legend(leg1_handles, leg1_labels, loc = 'upper left', bbox_to_anchor=(-0.15, 1.02), prop = fontP, handlelength = 2.5, handletextpad = 0.05, columnspacing = 0.1, ncol = 1, frameon = False) 

    # Label the value of the fixed variable 
    # above the legend 
    fixed_value = fixed_var[fixed_var_index] 
    if plotter_pars["fixed_variable"] == "temperature": 
        fixed_label = r"$\log_{10} (T_{\rm{init}} [\rm{K}]) = %.1f$" % (fixed_value, ) 
    elif plotter_pars["fixed_variable"] == "density": 
        fixed_label = r"$\log_{10} n_{\rm{H}} = %.1f$" % (fixed_value, ) 
    elif plotter_pars["fixed_variable"] == "metallicity": 
        fixed_label = r"$\log_{10} (Z / \rm{Z}_{\odot}) = %.1f$" % (fixed_value, ) 
    else: 
        raise Exception("ERROR: fixed_variable %s not recognised. Aborting" % (plotter_vars["fixed_variable"], )) 
    plt.title(fixed_label, fontsize = plotter_pars["legend_fontsize"]) 

    ax_dummy_leg2 = plt.axes([plotter_pars["subplot_right"] + 0.02, plotter_pars["subplot_bottom"], 0.98 - plotter_pars["subplot_right"], (plotter_pars["subplot_top"] - plotter_pars["subplot_bottom"]) / 2.0]) 
    ax_dummy_leg2.set_axis_off() 

    leg2_handles = [] 
    leg2_labels = [] 
    for i in range(plotter_pars["N_lines"]): 
        leg2_handles.append(matplotlib.lines.Line2D([0], [0], color = 'k', lw = plotter_pars["line_width"], linestyle = line_styles[i])) 

        line_value = np.log10(array_dimensions[plotter_pars["line_variable"]][line_var_indices[i]])
        if plotter_pars["line_variable"] == "density": 
            line_label = r"$\log_{10} n_{\rm{H}} = %.1f$" % (line_value, ) 
        elif plotter_pars["line_variable"] == "metallicity": 
            line_label = r"$\log_{10} (Z / \rm{Z}_{\odot}) = %.1f$" % (line_value, )

        leg2_labels.append(line_label) 

    plt.legend(leg2_handles, leg2_labels, loc = 'upper left', bbox_to_anchor=(-0.15, 1.0), prop = fontP, handlelength = 2.5, handletextpad = 0.05, columnspacing = 0.1, ncol = 1, frameon = False) 

    plt.savefig(plotter_pars["output_file"], dpi = 300)
    plt.close()

    return
