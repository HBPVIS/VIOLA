#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
E-I network connected with NEST topology
----------------------------------------

Simulation of a network consisting of an excitatory and an inhibitory
neuron population with distance-dependent connectivity.

The code bases on the script
    brunel_alpha_nest.py
which is part of NEST ( http://nest-initiative.org/ )
implementing a random balanced network (with alpha-shaped synapses)
as in

Brunel N, Dynamics of Sparsely Connected Networks of Excitatory and
Inhibitory Spiking Neurons, Journal of Computational Neuroscience 8,
183–208 (2000).

In contrast to the original version which does not take network
geometry into account, distance-dependent connections are here established
using the NEST topology module and a spatially confined external stimulus is
added.

The script writes to the output folder 'out_raw':
- neuron positions
- population GIDs
- plot of spike raster

- raw spike data in .gdf format for VIOLA
- configuration file (for raw data) for VIOLA

Usage:
::

    python topo_brunel_alpha_nest.py out_raw
'''

'''
Importing all necessary modules for simulation, analysis and plotting.
'''

import sys
# JURECA: remove global matplotlib from path such that local install can be found
try:
    sys.path.remove('/usr/local/software/jureca/Stages/2016a/software/SciPy-Stack/2016a-intel-para-2016a-Python-2.7.11/lib/python2.7/site-packages/matplotlib-1.5.1-py2.7-linux-x86_64.egg')
except (ValueError, KeyError) as err:
    pass
import os
import time
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
#from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

import scipy
from scipy.optimize import fsolve

import numpy as np
from numpy import exp, random, zeros_like, r_
from multiprocessing import cpu_count

import json

import nest
nest.set_verbosity('M_WARNING')
import nest.topology as tp

print('matplotlib version: ' + matplotlib.__version__)
print('numpy version: ' + np.__version__)
print('scipy version: ' + scipy.__version__)


plt.rcParams.update({
    'axes.xmargin': 0.0,
    'axes.ymargin': 0.0,
})

random.seed(123456)

# base color definitions
hex_col_ex = '#595289'   # blue with pastel character
hex_col_in = '#af143c'   # red with pastel character
hex_col_stim = '#696969' # DimGray

'''
Assigning the current time to a variable in order to determine the
build time of the network.
'''

startbuild = time.time()

'''
Assigning the simulation parameters to variables.
'''

dt      = 0.1    # Simulation time resolution in ms
simtime = 1500.  # Simulation time in ms
transient = 500. # Simulation transient, discarding spikes at times < transient

'''
Definition of the parameters crucial for the network state.
'''

g       = 4.5  # ratio inhibitory weight/excitatory weight (before: 5.0)
eta     = 2.   # external rate relative to threshold rate
epsilon = 0.1  # connection probability

'''
Definition of the number of neurons in the network.
'''

order     = 5000    # (before: 2500)
NE        = 4*order # number of excitatory neurons
NI        = 1*order # number of inhibitory neurons
N_neurons = NE+NI   # number of neurons in total

'''
Definition of connectivity parameters.
'''

CE    = int(epsilon*NE) # number of excitatory synapses per neuron
CI    = int(epsilon*NI) # number of inhibitory synapses per neuron
C_tot = int(CI+CE)      # total number of synapses per neuron

'''
Initialization of the parameters of the integrate-and-fire neurons and
the synapses. The parameters of the neuron are stored in a dictionary.
The synaptic currents are normalized such that the amplitude of the
PSP is J.
'''

tauSyn = 0.5  # synaptic time constant in ms
tauMem = 20.  # time constant of membrane potential in ms
CMem   = 100. # capacitance of membrane in in pF
theta  = 20.  # membrane threshold potential in mV
tRef   = 2.   # refractory period in ms
neuron_params= {"C_m":        CMem,
                "tau_m":      tauMem,
                "tau_syn_ex": tauSyn,
                "tau_syn_in": tauSyn,
                "t_ref":      tRef,
                "E_L":        0.,
                "V_reset":    0.,
                "V_m":        0.,
                "V_th":       theta}

J_ex = 40.     # postsynaptic amplitude in pA (before: 0.1 mV, converted to pA)
J_in = -g*J_ex # amplitude of inhibitory postsynaptic current

'''
Definition of the threshold rate, which is the external rate needed to fix
the membrane potential around its threshold (assuming just one external
connection). The rate of the Poisson generator is given in Hz. It is the
threshold rate multiplied with the relative rate eta.
'''

nu_th  = (theta * CMem) / (J_ex*exp(1)*tauMem*tauSyn)
p_rate = eta * nu_th * 1000.

'''
Parameters for a spatially confined stimulus.
'''

stim_radius = 0.5      # radius of a circle in mm for location of stimulus
mask_radius_stim = 0.1 # mask radius of stimulus in mm around each parrot neuron
num_stim_conn = 300    # number of connections inside mask_radius_stim
stim_start = 1000.     # start time of stimulus in ms
stim_duration = 50.    # duration of the stimulus onset in ms
stim_rate = 300.       # rate of parrot neurons in Hz during stimulus activation

'''
Definition of topology-specific parameters. Connection routines use fixed
indegrees = convergent connections with a fixed number of connections.
'''

extent_length = 4. # in mm (layer size = extent_length x extent_length)
sigma_ex = 0.3     # width of Gaussian profile for excitatory connections in mm
sigma_in = 0.3     # width of Gaussian profile for inhibitory connections in mm

delay_ex_c = 0.5 # constant term for linear distance-dep. delay in mm (exc.)
delay_ex_a = 0.5 # term for delay in mm
delay_in_c = 0.5 # term for linear distance-dep. delay in mm (inh.)
delay_in_a = 0.5 # term for delay in mm

delay_stim = 0.5 # delay between Poisson input to stimulus, and stimulus and exc.

pos_ex = list(((random.rand(2*NE) - 0.5) * extent_length).reshape(-1, 2))
pos_in = list(((random.rand(2*NI) - 0.5) * extent_length).reshape(-1, 2))

layerdict_ex = {
    'extent' : [extent_length, extent_length],
    'positions' : pos_ex,
    'elements' : 'iaf_psc_alpha',
    'edge_wrap' : True, # PBC
}

layerdict_in = {
    'extent' : [extent_length, extent_length],
    'positions' : pos_in,
    'elements' : 'iaf_psc_alpha',
    'edge_wrap' : True,
}

'''
The number of parrot neurons for the stimulus is computed by preserving the
density of excitatory neurons. The parrot neurons are placed inside a circle
around the center of the sheet.
'''

N_stim_square = int(NE * (2.*stim_radius)**2/extent_length**2)
pos_stim_square = list(((random.rand(2*N_stim_square) - 0.5) * 2.*stim_radius).reshape(-1, 2))

# discard those positions which do not fall into circle
pos_stim = []
for pos in pos_stim_square:
    if pos[0]**2 + pos[1]**2 <= stim_radius**2:
        pos_stim.append(pos)
N_stim = len(pos_stim)

layerdict_stim = {
    'extent' : [extent_length, extent_length],
    'positions' : pos_stim,
    'elements' : 'parrot_neuron',
    'edge_wrap' : True,
}

'''
Connection dictionaries are defined.
'''

conn_dict_ex = {
    'connection_type': 'convergent',
    'allow_autapses': False,
    'allow_multapses': True,
    'weights' : J_ex,
    'delays' : {
        'linear' : { # p(d) = c + a * d, d is distance
            'c' : delay_ex_c,
            'a' : delay_ex_a,
            }
        },
    'kernel' : {
        'gaussian' : {
            'p_center' : 1.,
            'sigma' : sigma_ex,
            'mean' : 0.,
            'c' : 0.,
            }
        },
    'number_of_connections' : CE,
    }

conn_dict_in = {
    'connection_type': 'convergent',
    'allow_autapses': False,
    'allow_multapses': True,
    'weights' : J_in,
    'delays' : {
        'linear' : {
            'c' : delay_in_c,
            'a' : delay_in_a,
            }
        },
    'kernel' : {
        'gaussian' : {
            'p_center' : 1.,
            'sigma' : sigma_in,
            'mean' : 0.,
            'c' : 0.,
            }
        },
    'number_of_connections' : CI,
    }

conn_dict_stim = {
    'connection_type': 'divergent',
    'weights' : J_ex,
    'delays' : delay_stim,
    'mask' : {
        'circular' : {
            'radius' : mask_radius_stim
            }
        },
    'kernel' : 1.,
    'number_of_connections' : num_stim_conn,
    }


def cmap_white_to_color(hexcolor, num, whitemin=True):
    '''
    Create linear colormap.
    '''
    rgb = mpc.hex2color(hexcolor)

    rs = np.linspace(1, rgb[0], num)
    gs = np.linspace(1, rgb[1], num)
    bs = np.linspace(1, rgb[2], num)

    rgbs = zip(rs, gs, bs)
    if not whitemin:
        rgbs = rgbs[::-1] # switch order of colors
    cmap = mpc.ListedColormap(tuple(rgbs))
    return cmap


if __name__ == '__main__':
    
    '''
    Destination for spike output and definition of file prefixes.
    '''
    
    if len(sys.argv) != 2:
        spike_output_path = 'out_raw'
    else:
        spike_output_path = sys.argv[-1]
    label = 'spikes' # spike detectors
    label_positions = 'neuron_positions' # neuron positions
    
    '''
    Create the file output destination folder if it does not exist.
    Delete old simulation files if the folder is already present
    '''
    
    if not os.path.isdir(spike_output_path):
        os.mkdir(spike_output_path)
    else:
        for fil in os.listdir(spike_output_path):
            os.remove(os.path.join(spike_output_path, fil))
    
    '''
    Reset the simulation kernel.
    Configuration of the simulation kernel by the previously defined time
    resolution used in the simulation. Setting "print_time" to True prints
    the already processed simulation time as well as its percentage of the
    total simulation time.
    '''
    
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": dt,
                          "print_time": False,
                          "overwrite_files": True,
                          'local_num_threads': cpu_count(),
                          'grng_seed': 234567})
    print('total_num_virtual_procs: ' + str(nest.GetKernelStatus('total_num_virtual_procs')))

    print("Building network")
    
    '''
    Configuration of the model `iaf_psc_alpha` and `poisson_generator`
    using SetDefaults(). This function expects the model to be the
    inserted as a string and the parameter to be specified in a
    dictionary. All instances of theses models created after this point
    will have the properties specified in the dictionary by default.
    '''
    
    nest.SetDefaults("iaf_psc_alpha", neuron_params)
    
    '''
    Creation of the topology layers for excitatory and inhibitory neurons.
    GIDs and neuron positions are written to file.
    '''
    
    layer_in = tp.CreateLayer(layerdict_in)
    layer_ex = tp.CreateLayer(layerdict_ex)
    layer_stim = tp.CreateLayer(layerdict_stim)
    
    tp.DumpLayerNodes(layer_ex, os.path.join(spike_output_path,
                                             label_positions + '-0.dat'))
    tp.DumpLayerNodes(layer_in, os.path.join(spike_output_path,
                                             label_positions + '-1.dat'))
    tp.DumpLayerNodes(layer_stim, os.path.join(spike_output_path,
                                             label_positions + '-2.dat'))
    
    nodes_ex = nest.GetChildren(layer_ex)[0] # nodes of ex/in neurons
    nodes_in = nest.GetChildren(layer_in)[0]
    nodes_stim = nest.GetChildren(layer_stim)[0]
    
    '''
    Distribute initial membrane voltages.
    '''

    for neurons in [nodes_ex, nodes_in]:
        for neuron in neurons:
            nest.SetStatus([neuron], {'V_m': theta * random.rand()})

    '''
    Create spike detectors for recording from the excitatory and the
    inhibitory populations and a poisson generator as noise source.
    The spike detectors are configured for writing to file.
    '''
    
    espikes = nest.Create("spike_detector")
    ispikes = nest.Create("spike_detector")
    stim_spikes = nest.Create("spike_detector")
    
    nest.SetStatus(espikes,[{
                       "label": os.path.join(spike_output_path, label + "-0"),
                       "withtime": True,
                       "withgid": True,
                       "to_file": True,
                       "start" : transient,
                       }])
    
    nest.SetStatus(ispikes,[{
                       "label": os.path.join(spike_output_path, label + "-1"),
                       "withtime": True,
                       "withgid": True,
                       "to_file": True,
                       "start" : transient,
                       }])
    
    nest.SetStatus(stim_spikes,[{
                       "label": os.path.join(spike_output_path, label + "-2"),
                       "withtime": True,
                       "withgid": True,
                       "to_file": True,
                       "start" : transient,
                       }])
    
    noise = nest.Create("poisson_generator", 1, {"rate": p_rate})
    
    '''
    External stimulus.
    '''
    
    pg_stim = nest.Create('poisson_generator', 1,
                          {'start': stim_start,
                           'stop': stim_start + stim_duration,
                           'rate': stim_rate})
    
    print("Connecting devices")
    
    '''
    Definition of a synapse using `CopyModel`, which expects the model
    name of a pre-defined synapse, the name of the customary synapse and
    an optional parameter dictionary. The parameters defined in the
    dictionary will be the default parameter for the customary
    synapse. Here we define one synapse for the excitatory and one for the
    inhibitory connections giving the previously defined weights
    '''
    
    nest.CopyModel("static_synapse","excitatory",{"weight":J_ex})
    nest.CopyModel("static_synapse","inhibitory",{"weight":J_in})
    
    '''
    Connecting the previously defined poisson generator to the excitatory
    and inhibitory neurons using the excitatory synapse. Since the poisson
    generator is connected to all neurons in the population the default
    rule ('all_to_all') of Connect() is used. The synaptic properties are
    inserted via syn_spec which expects a dictionary when defining
    multiple variables or a string when simply using a pre-defined
    synapse.
    '''
    
    nest.Connect(noise, nodes_ex, syn_spec="excitatory")
    nest.Connect(noise, nodes_in, syn_spec="excitatory")
    
    '''
    Connecting the excitatory, inhibitory and stimulus populations to the associated
    spike detectors using excitatory synapses. Here the same shortcut for the
    specification of the synapse as defined above is used.
    '''
    
    nest.Connect(nodes_ex, espikes, syn_spec="excitatory")
    nest.Connect(nodes_in, ispikes, syn_spec="excitatory")
    
    nest.Connect(nodes_stim, stim_spikes, syn_spec="excitatory")
    
    print("Connecting network")
    
    '''
    Connecting the excitatory and inhibitory populations using the
    pre-defined excitatory/inhibitory synapse and the connection dictionaries.
    First, update the connection dictionaries with the synapses.
    '''
    
    conn_dict_ex['synapse_model'] = 'excitatory'
    conn_dict_in['synapse_model'] = 'inhibitory'
    conn_dict_stim['synapse_model'] = 'excitatory'
    
    print("Excitatory connections")
    
    tp.ConnectLayers(layer_ex, layer_ex, conn_dict_ex)
    tp.ConnectLayers(layer_ex, layer_in, conn_dict_ex)
    
    print("Inhibitory connections")
    tp.ConnectLayers(layer_in, layer_ex, conn_dict_in)
    tp.ConnectLayers(layer_in, layer_in, conn_dict_in)
    
    '''
    Connect spike generator of external stimulus with the excitatory neurons.
    '''
    
    tp.ConnectLayers(layer_stim, layer_ex, conn_dict_stim)
    
    nest.Connect(pg_stim, nodes_stim, syn_spec={'weight': J_ex})
    
    '''
    Storage of the time point after the buildup of the network in a
    variable.
    '''

    endbuild=time.time()
    
    # # ConnPlotter test plot
    # if True:
    #     import ConnPlotter as cpl
    #     nest.CopyModel("static_synapse","STIM", {"weight":stim_weight_scale*J_ex})
    #     conn_dict_stim['synapse_model'] = 'STIM' # somehow 
    #     lList = [
    #         ('STIM', layerdict_stim),
    #         ('EX', layerdict_ex),
    #         ('IN', layerdict_in),
    #         ]
    #     cList = [
    #         ('STIM', 'EX', conn_dict_stim),
    #         ('EX', 'EX', conn_dict_ex),
    #         ('EX', 'IN', conn_dict_ex),
    #         ('IN', 'EX', conn_dict_in),
    #         ('IN', 'IN', conn_dict_in),
    #         ]
    #     synTypes = ((
    #         cpl.SynType('excitatory', J_ex, 'r'),
    #         cpl.SynType('inhibitory', J_in, 'b'),
    #         cpl.SynType('STIM', stim_weight_scale*J_ex, 'k')
    #         ),)
    #     s_cp = cpl.ConnectionPattern(lList, cList, synTypes=synTypes)
    #     s_cp.plot(colorLimits=[0,100])
    #     s_cp.plot(aggrSyns=True, colorLimits=[0,100])
    #     plt.show()
    
    
    
    '''
    Simulation of the network.
    '''
    
    print("Simulating")
    
    nest.Simulate(simtime)
    
    '''
    Storage of the time point after the simulation of the network in a
    variable.
    '''
    
    endsimulate= time.time()
    
    '''
    Reading out the total number of spikes received from the spike
    detector connected to the excitatory population and the inhibitory
    population.
    '''
    
    events_ex = nest.GetStatus(espikes,"n_events")[0]
    events_in = nest.GetStatus(ispikes,"n_events")[0]
    events_stim = nest.GetStatus(stim_spikes,"n_events")[0]
    
    '''
    Calculation of the average firing rate of the excitatory and the
    inhibitory neurons by the simulation time. The
    multiplication by 1000.0 converts the unit 1/ms to 1/s=Hz.
    '''
    
    rate_ex   = events_ex/(simtime-transient)*1000./len(nodes_ex)
    rate_in   = events_in/(simtime-transient)*1000./len(nodes_in)
    rate_stim   = events_stim/(simtime-transient)*1000./len(nodes_in)
    
    '''
    Reading out the number of connections established using the excitatory
    and inhibitory synapse model. The numbers are summed up resulting in
    the total number of synapses.
    '''
    
    num_synapses = nest.GetDefaults("excitatory")["num_connections"]+\
                   nest.GetDefaults("inhibitory")["num_connections"]
    
    '''
    Establishing the time it took to build and simulate the network by
    taking the difference of the pre-defined time variables.
    '''
    
    build_time = endbuild-startbuild
    sim_time   = endsimulate-endbuild
    
    '''
    Printing the network properties, firing rates and building times.
    '''
    
    print("Brunel network simulation (Python)")
    print("Number of neurons : {0}".format(N_neurons))
    # including devices and noise
    print("Number of synapses: {0}".format(num_synapses))
    # neurons + noise + spike detectors
    print("       Exitatory  : {0}".format(int(CE * N_neurons) + 2 * N_neurons))
    print("       Inhibitory : {0}".format(int(CI * N_neurons)))
    print("Excitatory rate   : %.2f Hz" % rate_ex)
    print("Inhibitory rate   : %.2f Hz" % rate_in)
    print("Stimulus rate     : %.2f Hz" % rate_stim)
    print("Building time     : %.2f s" % build_time)
    print("Simulation time   : %.2f s" % sim_time)

    nest.sli_run('memory_thisjob') # virtual memory size of NEST process
    memory = nest.sli_pop()
    print("Memory            : %.2f kB" % memory)
    
    '''
    A dictionary for population parameters is created to allow for easier access.
    '''
    
    pops = {}
    pops['EX'] = {}
    pops['IN'] = {}
    pops['STIM'] = {}
    
    # neuron numbers
    pops['EX']['N'] = NE
    pops['IN']['N'] = NI
    pops['STIM']['N'] = N_stim
    
    # positions
    pops['EX']['pos'] = pos_ex
    pops['IN']['pos'] = pos_in
    pops['STIM']['pos'] = pos_stim
    
    # layer
    pops['EX']['layer'] = layer_ex
    pops['IN']['layer'] = layer_in
    pops['STIM']['layer'] = layer_stim
    
    # layerdict
    pops['EX']['layerdict'] = layerdict_ex
    pops['IN']['layerdict'] = layerdict_in
    pops['STIM']['layerdict'] = layerdict_stim
    
    # nodes
    pops['EX']['nodes'] = nodes_ex
    pops['IN']['nodes'] = nodes_in
    pops['STIM']['nodes'] = nodes_stim
    
    # rate
    pops['EX']['rate'] = rate_ex
    pops['IN']['rate'] = rate_in
    pops['STIM']['rate'] = rate_stim
    
    # events
    pops['EX']['events'] = nest.GetStatus(espikes, 'events')[0]
    pops['IN']['events'] = nest.GetStatus(ispikes, 'events')[0]
    pops['STIM']['events'] = nest.GetStatus(stim_spikes, 'events')[0]
    
    # population colors
    pops['EX']['color'] =  mpc.hex2color(hex_col_ex)
    pops['IN']['color'] =  mpc.hex2color(hex_col_in)
    pops['STIM']['color'] =  mpc.hex2color(hex_col_stim)
    
    # dark connection colors (just darker than population colors)
    pops['EX']['conn_color_dark'] = tuple(np.array(pops['EX']['color']) * 0.9) # darken
    pops['IN']['conn_color_dark'] = tuple(np.array(pops['IN']['color']) * 0.9)
    pops['STIM']['conn_color_dark'] = tuple(np.array(pops['EX']['color']) * 0.9)

    # light connection colors (just lighter than population colors, note: <1)
    pops['EX']['conn_color_light'] = tuple(np.array(pops['EX']['color']) * 1.4) # lighten
    pops['IN']['conn_color_light'] = tuple(np.array(pops['IN']['color']) * 1.4)
    pops['STIM']['conn_color_light'] = tuple(np.array(pops['EX']['color']) * 1.4)
    
    # targets of the neuron type
    pops['EX']['tgts'] = ['EX', 'IN']
    pops['IN']['tgts'] = ['EX', 'IN']
    pops['STIM']['tgts'] = ['EX']


'''
In the following, functions for rudimentary postprocessing are defined.
They are called at the bottom of the script.
First, spike files have to be merged and the population GIDs and configuration
files for VIOLA are written to file.
'''

def merge_spike_files():
    '''
    Merges spike files from different threads.
    '''
    print("Merging spike files")
    for i, pop in enumerate(['EX', 'IN', 'STIM']):
        old_filenames = glob.glob(os.path.join(spike_output_path, label + '-' + str(i) + '*.gdf'))
        data = np.empty((0, 2))
        for t in range(len(old_filenames)):
            data = np.vstack([data, np.loadtxt(old_filenames[t])])
            os.remove(old_filenames[t])
        order = np.argsort(data[:, 1]) # sort spike times
        data = data[order]
        # write to new file having the same filename as for thread 0
        new_filename = os.path.join(spike_output_path, label+'-'+ str(i) + '.gdf')
        with open(new_filename, 'w') as f:
            for line in data:
                f.write('%d\t%.3f\n' % (line[0], line[1]))
        f.close()
    return

def write_population_GIDs():
    '''
    Writes first and last neuron GID of both poulations to file.
    '''
    print("Writing population GIDs")
    fname = os.path.join(spike_output_path, 'population_GIDs.dat')
    with open(fname, 'w') as f:
        f.write('%d\t%d\n' % (nodes_ex[0], nodes_ex[-1]))
        f.write('%d\t%d\n' % (nodes_in[0], nodes_in[-1]))
        f.write('%d\t%d\n' % (nodes_stim[0], nodes_stim[-1]))
    f.close()
    return


'''
A configuration file for VIOLA.
'''

def create_viola_config_raw():
    '''
    Creates a configuration file for the visualization of raw simulation output
    with VIOLA.
    '''
    # hex colors for VIOLA
    popColors = []
    for pop in ['EX', 'IN', 'STIM']:
        popColors.append(mpc.rgb2hex(pops[pop]['color']))

        # configuration dictionary for VIOLA
        config_dict = {}
        config_dict.update({
            "popNum": 3,
            "popNames": ','.join(['EX', 'IN', 'STIM']),
            "spikesFiles": [label+'-%i.gdf' % X for X in [0,1,2]],
            "timestamps": int(simtime / dt),
            "resolution": dt,
            "xSize": extent_length,
            "ySize": extent_length,
            "dataType": "neuron",
            "posFiles": [label_positions+'-%i.dat' % X for X in [0,1,2]],
            "timelineLength": 100,
            "popColors": popColors,
        })

    with open(os.path.join(spike_output_path, 'config_raw.json'), 'w') as f:
        json.dump(config_dict, f)


'''
Plotting functions for a network sketch and a ConnPlotter variant.
'''

def figure_network_sketch():
    '''
    Plots a network sketch and a illustrates connectivity using
    ConnPlotter's style.
    '''

    print('Plotting network sketch')
    red_conn_dens = 1 # show connections in steps of
    dilute_neurons = 1 # show neurons in steps of
    print('  Diluting connection density: {}'.format(red_conn_dens))
    print('  Diluting number of neurons shown: {}'.format(dilute_neurons))

    # set up figure
    fig = plt.figure(figsize=(13,5))
    # grid spec for left and right panel
    gs1 = gridspec.GridSpec(1, 10, wspace=0.0, left=0.05, right=1.05, bottom=0., top=1)
    ax1 = plt.subplot(gs1[0, 3:], projection='3d')
    

    # plot connectivity using ConnPlotter's style
    gs = gridspec.GridSpec(1, 3, wspace=0.5, left=0.05, right=1.)
    gs0 = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[0,0],
                                           wspace=0.01)

    lList = [
        ('STIM', layerdict_stim),
        ('EX', layerdict_ex),
        ('IN', layerdict_in),
        ]
    conns = [[1, 0], [1, 1], [1, 1]]
    pList = ['EX', 'IN']
    cDicts = [conn_dict_stim, conn_dict_ex, conn_dict_in]
    for i, ((pre, lDict), cDict, conn) in enumerate(zip(lList, cDicts, conns)):
        for j, post in enumerate(pList):
            ax = fig.add_subplot(gs0[i, j], aspect='equal')
            extent = lDict['extent']
            x = np.linspace(-extent[0]/2, extent[0]/2, 51) # changed from 101
            y = np.linspace(-extent[1]/2, extent[1]/2, 51)
            X,Y = np.meshgrid(x, y)
            C = np.zeros(X.shape)
            if conn[j]:
                if 'kernel' not in cDict.keys() or cDict['kernel'] == 1.:
                    try:
                        weights = epsilon_stim = num_stim_conn*N_stim / NE * cDict['weights']
                        C[np.sqrt(X**2 + Y**2) <= cDict['mask']['circular']['radius']] = weights
                        # cmap = 'gray_r'
                        colors = [(1, 1, 1), (0, 0, 1)]
                        cmap = cmap_white_to_color(hex_col_ex, 64)
                        #cmap =  LinearSegmentedColormap.from_list('reds', colors, N=64)
                        vmin = 0
                        vmax = weights
                    except KeyError as ae:
                        raise ae
                elif type(cDict['kernel']) is dict:
                    try:
                        sigma = cDict['kernel']['gaussian']['sigma']
                        if 'mask' in cDict.keys():
                            mask = np.sqrt(X**2 + Y**2) <= cDict['mask']['circular']['radius']
                            weights = cDict['weights']*epsilon
                            C[mask] = weights*np.exp(-(X[mask]**2 + Y[mask]**2) / (2*sigma**2)) # / (2*np.pi*sigma**2)
                            if weights > 0:
                                colors = [(1, 1, 1), (0, 0, 1)]
                                cmap = cmap_white_to_color(hex_col_ex, 64)
                                #cmap =  LinearSegmentedColormap.from_list('blues', colors, N=64)
                                vmin = 0
                                vmax = weights
                            else:
                                colors = [(1, 0, 0), (1, 1, 1)]
                                cmap = cmap_white_to_color(hex_col_in, 64)
                                #cmap =  LinearSegmentedColormap.from_list('reds', colors, N=64)
                                vmin = weights
                                vmax = 0
                        else:
                            weights = cDict['weights']
                            C = weights*np.exp(-(X**2 + Y**2) / (2*sigma**2)) # / (2*np.pi*sigma**2)
                            if weights > 0:
                                colors = [(1, 1, 1), (0, 0, 1)]
                                cmap = cmap_white_to_color(hex_col_ex, 64)
                                #cmap =  LinearSegmentedColormap.from_list('blues', colors, N=64)
                                vmin = 0
                                vmax = weights
                            else:
                                colors = [(1, 0, 0), (1, 1, 1)]
                                cmap = cmap_white_to_color(hex_col_in, 64, whitemin=False)
                                #cmap =  LinearSegmentedColormap.from_list('reds', colors, N=64)
                                vmin = weights
                                vmax = 0
                    except KeyError as ae:
                        raise ae
                else:
                    pass

            cmap.set_bad('0.75')
            im = ax.pcolormesh(X,Y,np.ma.array(C, mask=C==0), cmap=cmap, vmin=vmin, vmax=vmax)
            # im = ax.pcolormesh(X,Y,C, cmap=cmap, vmin=vmin, vmax=vmax)

            if j == (len(pList)-1):
                bbox = np.array(ax.get_position())
                cax = fig.add_axes([bbox[1][0]+0.01, bbox[0][1], 0.015, (bbox[1][1]-bbox[0][1])])
                axcb = fig.colorbar(im, cax=cax, orientation='vertical')
                cbarlabel = r'$\epsilon_{YX}Jg_{YX}$ (pA)'
                # cbarlabel = r'$\epsilon_{Y,\mathrm{%s}}Jg_{Y,\mathrm{%s}}$ (pA)' % (pre,pre)
                axcb.set_label(cbarlabel)
                axcb.locator = MaxNLocator(nbins=5)
                axcb.update_ticks()

            ax.set_xticks([-2., -1, 0, 1., 2.])
            ax.set_yticks([-2., -1, 0, 1., 2.])

            if i != (len(lList)-1):
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r'$x_i - x_j$ (mm)', labelpad=0)
            if i == 0:
                ax.set_title(r'$Y=${}'.format(post))
            if j != 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel('$X=${}\n'.format(pre) + r'$y_i - y_j$ (mm)', labelpad=0)

            if i == 0 and j == 0:
                ax.text(0.05, 0.95, 'A',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=16, fontweight='demibold',
                    transform=fig.transFigure)


    # network sketch
    ax1.text2D(0.4, 0.95, 'B',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='demibold',
        transform=fig.transFigure)

    # build figure from bottom to top
    pops_list = ['IN', 'EX', 'STIM'] # bottom, center, top
    dots_IN_IN, srcdot_IN_IN = plot_connections(ax1, 'IN', 'IN', pops_list, red_conn_dens)
    plot_layer(ax1, 'IN', pops_list, dilute_neurons)
    plot_dots(ax1, dots_IN_IN)
    plot_dots(ax1, srcdot_IN_IN)
    dots_IN_EX, srcdot_IN_EX = plot_connections(ax1, 'IN', 'EX', pops_list, red_conn_dens)
    _, srcdot_EX_IN = plot_connections(ax1, 'EX', 'IN', pops_list, red_conn_dens)
    dots_EX_EX, srcdot_EX_EX = plot_connections(ax1, 'EX', 'EX', pops_list, red_conn_dens)
    plot_layer(ax1, 'EX', pops_list, dilute_neurons)
    plot_dots(ax1, dots_IN_EX)
    plot_dots(ax1, dots_EX_EX)
    plot_dots(ax1, srcdot_IN_EX)
    plot_dots(ax1, srcdot_EX_EX)
    plot_dots(ax1, srcdot_EX_IN)
    _, srcdot_STIM_EX = plot_connections(ax1, 'STIM', 'EX', pops_list, red_conn_dens)
    plot_layer(ax1, 'STIM', pops_list, dilute_neurons)
    plot_dots(ax1, srcdot_STIM_EX)

    # make plot look nice
    ax1.set_xlabel('$x$ (mm)', labelpad=-1)
    ax1.set_ylabel('$y$ (mm)', labelpad=-1)
    ax1.set_xticks([-2., -1, 0, 1., 2.])
    ax1.set_yticks([-2., -1, 0, 1., 2.])
    ax1.set_xlim(-1.95, 1.95)
    ax1.set_ylim(-1.95, 1.95)
    ax1.xaxis.set_tick_params(pad=-1)
    ax1.yaxis.set_tick_params(pad=-1)
    ax1.w_zaxis.line.set_lw(0.)
    ax1.set_zticks([])

    ax1.grid(False)
    ax1.xaxis.pane.set_edgecolor('white')
    ax1.yaxis.pane.set_edgecolor('white')
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    # legend
    handles = \
        [Patch(color=pops['STIM']['color']),
         Patch(color=pops['EX']['color']),
         Patch(color=pops['IN']['color']),
         plt.Line2D((0,1),(0,0), color='white', marker='o',
                    markeredgecolor='black', linestyle=''),
         plt.Line2D((0,1),(0,0), color=pops['EX']['conn_color_light']),
         plt.Line2D((0,1),(0,0), color=pops['IN']['conn_color_light'])]
    labels = \
        ['STIM',
         'EX',
         'IN',
         'source',
         'exc. connection',
         'inh. connection']
    ax1.legend(handles, labels, numpoints=1, loc=2, bbox_to_anchor=(0.7, 0.9),
              fontsize=10)

    ax1.view_init(elev=20, azim=-60)

    fig.savefig(os.path.join(spike_output_path, 'network_sketch.pdf'), dpi=320,
                bbox_inches=0)
    fig.savefig(os.path.join(spike_output_path, 'network_sketch.eps'), dpi=320,
                bbox_inches=0)

'''
Definition of helper functions for the network sketch.
'''

def plot_layer(ax, pop, pops_list, dilute_neurons):
    # plot neurons at their original location
    pos = np.array(pops[pop]['pos']).transpose()
    z0 = pops_list.index(pop)

    xshow = pos[0][0:len(pos[0]):dilute_neurons]
    yshow = pos[1][0:len(pos[1]):dilute_neurons]

    ax.plot(xshow, yshow, zs=z0,
            marker=',',
            linestyle='None',
            color=pops[pop]['color'],
            alpha=1.)
    ax.text(-2, -2.8, z0+0.3, pop)
    return


def plot_connections(ax, src, tgt, pops_list, red_conn_dens):

    # z-positions
    z0 = pops_list.index(src)
    z1 = z0 + (pops_list.index(tgt) - z0)

    # x,y-positions
    if src == 'STIM':
        xyloc = [0., 0.]
    elif src == tgt:
        xyloc = [0.8, 0.8]
    elif src == 'EX':
        xyloc = [0.8, -0.8]
    elif src == 'IN':
        xyloc = [-0.8, -0.8]

    srcid = tp.FindNearestElement(pops[src]['layer'], xyloc, False)
    srcloc = tp.GetPosition(srcid)[0]
    tgtsloc = np.array(tp.GetTargetPositions(srcid,
                                             pops[tgt]['layer'])[0])
    # targets do not get picked in the same order;
    # they are sorted here for reproducibility
    tgtsloc = tgtsloc[np.argsort(tgtsloc[:,0])]
    tgtsloc_show = tgtsloc[0:len(tgtsloc):red_conn_dens]
    for tgtloc in tgtsloc_show:
        ax.plot([srcloc[0], tgtloc[0]], [srcloc[1], tgtloc[1]],
                [z0, z1], c=pops[src]['conn_color_light'], linewidth=1.)
        # highlight target
        ax.plot(xs=[tgtloc[0]], ys=[tgtloc[1]], zs=[z1],
                marker='o',
                markeredgecolor='none',
                markersize=2, color=pops[src]['conn_color_dark'],
                alpha=1.)

    # to be printed on top
    dots = [tgtsloc_show, z1, pops[src]['conn_color_dark'], 'none', 2]
    srcdot = [srcloc, z0, 'white', 'black', 3]
    return dots, srcdot


def plot_dots(ax, dots):
    if type(dots[0][0]) == np.ndarray:
        xs = zip(*dots[0])[0]
        ys = zip(*dots[0])[1]
    else:
        xs = [dots[0][0]]
        ys = [dots[0][1]]
    ax.plot(xs, ys, zs=dots[1], marker='o', markeredgecolor=dots[3],
            markersize=dots[4], c=dots[2], linestyle='none', alpha=1.)
    return


'''
Plot a figure of spiking activity showing unsorted and sorted raster plots and
spike counts.
'''

def figure_raster(times):
    print('Plotting spiking activity for time interval (ms): ' + str(times))

    # stepsize for diluting (1 = all)
    dilute = int(5) # int
    print('  Diluting spike number: {}'.format(dilute))


    fig = plt.figure(figsize=(13., 8.))
    fig.subplots_adjust(top=0.94, bottom=0.1, left=0.08, right=0.97,
                        wspace=0.3, hspace=1.)
    gs = gridspec.GridSpec(6,5)


    # A: unsorted raster
    gs_cell = gs[:2, :4]
    pops_list = ['STIM', 'EX', 'IN'] # top to bottom
    ax0 = _plot_raster_unsorted('A', gs_cell, pops_list, times, dilute)


    # B: spike count histogram over unit
    gs_cell = gs[:2, 4]
    pops_list = ['STIM', 'EX', 'IN']
    _plot_unit_histogram('B', gs_cell, pops_list, sharey=ax0)


    # C: spike count histogram over time
    gs_cell = gs[2:4, :4]
    pops_list = ['STIM', 'EX', 'IN'] # top to bottom
    _plot_time_histogram('C', gs_cell, pops_list, times)


    # legend to the bottom right
    ax = plt.subplot(gs[2:4,4:]) # just for the legend
    plt.axis('off')
    handles = [Patch(color=pops['STIM']['color']),
               Patch(color=pops['EX']['color']),
               Patch(color=pops['IN']['color'])]
    labels = ['STIM',
              'EX',
              'IN']
    ax.legend(handles, labels, loc='center')


    # D: sorted raster
    gs_cell = gs[4:6, :4]
    pops_list = ['EX', 'IN', 'STIM']
    _plot_raster_sorted('D', gs_cell, pops_list, times, dilute)


    # E: spike count histogram over space
    gs_cell = gs[4:6, 4]
    pops_list = ['IN', 'EX', 'STIM'] # bottom to top
    _plot_space_histogram('E', gs_cell, pops_list)




    fig.savefig(os.path.join(spike_output_path, 'raster.pdf'), dpi=320)
    fig.savefig(os.path.join(spike_output_path, 'raster.eps'), dpi=320)


'''
Definition of helper functions for the spiking activity.
'''

def _plot_spikes(ax, dilute, nodes,
                 events,
                 layerdict,
                 color='r',
                 marker=',', poplabel='EX',
                 position_sorted=True):
    '''
    Plots unsorted or sorted spike raster, flexible for both populations.
    '''
    X = []
    T = []
    for i, j in enumerate(nodes):
        # extract spikes
        t = events['times'][events['senders'] == j]
        x, y = layerdict['positions'][i]
        if t.size > 0:
            T = r_[T, t] # concatenate spike times

            if position_sorted:
                pos = x # sorted by x positions
            else:
                pos = j

            X = r_[X, zeros_like(t) + pos]

    ax.plot(T[::dilute], X[::dilute], marker, markersize=.1, color=color, label=poplabel,
                rasterized=False)
    return


def _plot_space_histogram(label, gs_cell, pops_list):
    gs_loc = gridspec.GridSpecFromSubplotSpec(1, 3, gs_cell, wspace=0.15)

    binsize = 0.1 # should be the same as used for preprocessing
    bins = np.arange(-2, 2+binsize, binsize)
    xlists = []
    for x, gid0, senders in zip([np.array(pops['IN']['layerdict']['positions'])[:, 0],
                                 np.array(pops['EX']['layerdict']['positions'])[:, 0],
                                 np.array(pops['STIM']['layerdict']['positions'])[:, 0]],
                                [pops['IN']['nodes'][0],
                                 pops['EX']['nodes'][0],
                                 pops['STIM']['nodes'][0]],
                                [pops['IN']['events']['senders'],
                                 pops['EX']['events']['senders'],
                                 pops['STIM']['events']['senders']]):
        xlists += [[x[n-gid0] for n in senders]]

    data = {}
    data['IN'] = xlists[0]
    data['EX'] = xlists[1]
    data['STIM'] = xlists[2]

    for i,pop in enumerate(pops_list):
        ax = plt.subplot(gs_loc[0,i])
        ax.hist(data[pop], bins=bins, histtype='stepfilled',
                color=pops[pop]['color'], edgecolor='none',
                orientation='horizontal')
        ax.set_ylim(bins[0], bins[-1])
        ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2, prune='upper'))
        plt.xticks(rotation=-90)

        if i==0:
            ax.text(-0.6, 1.05, label, ha='left', va='bottom', fontsize=16,
                    fontweight='demibold', transform=ax.transAxes)
        if i==int(len(pops_list)/2):
            ax.set_title('spike count\n' + r'($\Delta={}$ mm)'.format(binsize))
            ax.set_xlabel('count')
    return


def _plot_time_histogram(label, gs_cell, pops_list, times):
    gs_loc = gridspec.GridSpecFromSubplotSpec(3,1, gs_cell, hspace=0.15)
    # binsize should be the same as used for preprocessing
    binsize = 1 # in ms
    bins = np.arange(transient, simtime+binsize, binsize)
    for i,pop in enumerate(pops_list):
        ax = plt.subplot(gs_loc[i,0])
        ax.hist(pops[pop]['events']['times'], bins=bins, histtype='stepfilled',
                color=pops[pop]['color'], edgecolor='none')
        ax.set_ylim(bottom=0) # fixing only the bottom
        ax.set_xlim(times[0], times[1])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='upper'))

        if i==0:
            ax.set_title('spike count ' + r'($\Delta t={}$ ms)'.format(binsize))
            ax.text(-0.05, 1.05, label, ha='left', va='bottom', fontsize=16,
                    fontweight='demibold', transform=ax.transAxes)
        if i==int(len(pops_list)/2):
            ax.set_ylabel('count')
        else:
            ax.set_xticklabels([])
    return


def _plot_unit_histogram(label, gs_cell, pops_list, sharey):
    tot_neurons = 0
    ratios = []
    for pop in pops_list:
        tot_neurons += pops[pop]['N']
    for pop in pops_list:
        frac = 1.*pops[pop]['N']/tot_neurons
        if frac < 0.1:
            frac = 0.1
        ratios.append(frac)

    binsize = order / extent_length / 2. # neurons

    bins = np.arange(min([min(pops[pop]['nodes']) for pop in pops_list]),
                     max([max(pops[pop]['nodes']) for pop in pops_list])+binsize,
                     binsize)

    ax = plt.subplot(gs_cell)
    for i,pop in enumerate(pops_list):
        ax.hist(pops[pop]['events']['senders'],
                bins=bins, histtype='stepfilled',
                color=pops[pop]['color'], edgecolor='none',
                orientation='horizontal', stacked=False, alpha=1)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.axis(ax.axis('tight'))
    ax.set_xlabel('count')
    ax.set_ylim(sharey.get_ylim())
    ax.set_yticklabels([])

    ax.text(-0.18, 1.05, label, ha='left', fontsize=16, va='bottom',
            fontweight='demibold', transform=ax.transAxes)
    ax.set_title('spike count\n' + r'($\Delta={}$ units)'.format(int(binsize)))
    
    return


def _plot_raster_unsorted(label, gs_cell, pops_list, times, dilute):
    tot_neurons = 0
    ratios = []
    for pop in pops_list:
        tot_neurons += pops[pop]['N']
    for pop in pops_list:
        frac = 1.*pops[pop]['N']/tot_neurons
        if frac < 0.1:
            frac = 0.1
        ratios.append(frac)

    ax = plt.subplot(gs_cell)
    for i,pop in enumerate(pops_list):
        _plot_spikes(ax, dilute, nodes=pops[pop]['nodes'],
                     events=pops[pop]['events'],
                     layerdict=pops[pop]['layerdict'],
                     color=pops[pop]['color'],
                     marker=',', poplabel=pop,
                     position_sorted=False)
    ax.axis(ax.axis('tight'))
    ax.set_xlim(times[0], times[1])
    ax.set_xticklabels([])
    ax.set_xlabel('')

    min_node = np.min(pops[pop]['nodes'])
    max_node = np.max(pops[pop]['nodes'])

    ax.set_title('unsorted spike raster')
    ax.text(-0.05, 1.05, label, fontsize=16, ha='left',
            va='bottom', fontweight='demibold', transform=ax.transAxes)
    ax.set_ylabel('neuron ID')
        
    return ax


def _plot_raster_sorted(label, gs_cell, pops_list, times, dilute):
    ax = plt.subplot(gs_cell)
    for pop in pops_list:
        _plot_spikes(ax, dilute, nodes=pops[pop]['nodes'],
                     events=pops[pop]['events'],
                     layerdict=pops[pop]['layerdict'],
                     color=pops[pop]['color'],
                     marker=',', poplabel=pop,
                     position_sorted=True)

    ax.set_title('sorted spike raster')
    ax.set_ylabel('x position (mm)')
    ax.set_xlabel('time (ms)')
    ax.set_xlim(times[0], times[1])
    ax.text(-0.05, 1.05, label, ha='left', fontsize=16, va='bottom',
            fontweight='demibold', transform=ax.transAxes)
    return



if __name__=='__main__':
    # these functions are needed for generating test data for VIOLA
    merge_spike_files()
    write_population_GIDs()
    create_viola_config_raw()

    # these functions are optional
    if False:
        figure_network_sketch()

        times = [transient, simtime] # displayed time interval
        figure_raster(times)
