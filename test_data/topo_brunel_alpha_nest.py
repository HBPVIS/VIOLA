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

import matplotlib
matplotlib.use('Agg')

from scipy.optimize import fsolve

import nest
nest.set_verbosity('M_WARNING')
import nest.topology as tp

import time
import os
import sys
import glob
import numpy as np
from numpy import exp, random, zeros_like, r_
from multiprocessing import cpu_count

import json

import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

random.seed(123456)

'''
Definition of functions used in this example. First, define the
Lambert W function implemented in SLI. The second function
computes the maximum of the postsynaptic potential for a synaptic
input current of unit amplitude (1 pA) using the Lambert W
function. Thus function will later be used to calibrate the synaptic
weights.
'''

def LambertWm1(x):
    nest.sli_push(x); nest.sli_run('LambertWm1'); y=nest.sli_pop()
    return y

def ComputePSPnorm(tauMem, CMem, tauSyn):
  a = (tauMem / tauSyn)
  b = (1.0 / tauSyn - 1.0 / tauMem)

  # time of maximum
  t_max = 1.0/b * ( -LambertWm1(-exp(-1.0/a)/a) - 1.0/a )

  # maximum of PSP for current of unit amplitude
  return exp(1.0)/(tauSyn*CMem*b) * ((exp(-t_max/tauMem) - exp(-t_max/tauSyn)) / b - t_max*exp(-t_max/tauSyn))

'''
Assigning the current time to a variable in order to determine the
build time of the network.
'''

startbuild = time.time()

'''
Assigning the simulation parameters to variables.
'''

dt      = 0.1    # Simulation time resolution in ms
simtime = 2000.  # Simulation time in ms
transient = 500. # Simulation transient, discarding spikes at times < transient

'''
Definition of the parameters crucial for the network state.
'''

g       = 4.   # ratio inhibitory weight/excitatory weight (before: 5.0)
eta     = 2.   # external rate relative to threshold rate
epsilon = 0.1  # connection probability

'''
Definition of the number of neurons in the network.
'''

order     = 5000   # (before: 2500)
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

tauSyn = 0.5    # synaptic time constant in ms
tauMem = 20.    # time constant of membrane potential in ms
CMem   = 250.   # capacitance of membrane in in pF
theta  = 20.    # membrane threshold potential in mV
tRef = 2.       # refractory period in ms
neuron_params= {"C_m":        CMem,
                "tau_m":      tauMem,
                "tau_syn_ex": tauSyn,
                "tau_syn_in": tauSyn,
                "t_ref":      tRef,
                "E_L":        0.,
                "V_reset":    0.,
                "V_m":        0.,
                "V_th":       theta}
J      = 0.6        # postsyaptic amplitude in mV (before: 0.1)
J_unit = ComputePSPnorm(tauMem, CMem, tauSyn)
J_ex   = J / J_unit # amplitude of excitatory postsynaptic current
J_in   = -g*J_ex    # amplitude of inhibitory postsynaptic current

'''
Definition of the threshold rate, which is the external rate needed to fix
the membrane potential around its threshold, the external firing rate
and the rate of the Poisson generator which is multiplied by the
in-degree CE and converted to Hz by multiplication by 1000.
'''

nu_th  = (theta * CMem) / (J_ex*CE*exp(1)*tauMem*tauSyn)
nu_ex  = eta*nu_th
p_rate = 1000.*nu_ex*CE

'''
Parameters for a spatially confined stimulus.
'''

stim_radius = 0.2       # radius of a circle in mm for location of stimulus
mask_radius_stim = 0.1  # mask radius of stimulus in mm around each parrot neuron
num_stim_conn = 100     # number of connections inside mask_radius_stim
stim_start = 1650.      # start time of stimulus in ms
stim_duration = 5.      # duration of the stimulus onset in ms
stim_rate = 5000.       # rate of parrot neurons in Hz during stimulus activation

'''
Definition of topology-specific parameters. Connection routines use fixed
indegrees = convergent connections with a fixed number of connections.
'''

extent_length = 4.   # in mm (layer size = extent_length x extent_length)
sigma_ex = 0.25      # width of Gaussian profile for excitatory connections in mm
sigma_in = 0.3       # width of Gaussian profile for inhibitory connections in mm

delay_ex_c = 0.3     # constant term for linear distance-dependent delay in mm
delay_ex_a = 0.7     # linear term for delay in mm (for excitatory connections)
delay_in = 1.        # constant delay for inhibitory connections in mm

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
pos_stim_square = list(((random.rand(2*N_stim_square) - 0.5) * stim_radius).reshape(-1, 2))

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
        'constant' : {
            'value' : delay_in
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
    'delays' : dt,
    'mask' : {
        'circular' : {
            'radius' : mask_radius_stim
            }
        },
    'kernel' : 1.,
    'number_of_connections' : num_stim_conn,
    }


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
        nest.SetStatus([neuron], {'V_m': theta * np.random.rand()})

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

pg_stim = nest.Create('poisson_generator', 1, {'start': stim_start,
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
pops['EX']['color'] =  mpc.hex2color('#b22222')       # firebrick
pops['IN']['color'] =  mpc.hex2color('#0000cd')       # MediumBlue
pops['STIM']['color'] =  mpc.hex2color('#696969')     # DimGray

# population colors (just darker than population colors
pops['EX']['conn_color'] = mpc.hex2color('#ff3030')   # firebrick1
pops['IN']['conn_color'] = mpc.hex2color('#4169e1')   # RoyalBlue
pops['STIM']['conn_color'] = mpc.hex2color('#ff3030') # firebrick1

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
    red_conn_dens = 1 # reduce connection density
    print('  Diluting connection density: {}'.format(red_conn_dens))

    # set up figure
    fig = plt.figure(figsize=(6.5*2,5))
    # fig.subplots_adjust(top=1., bottom=0.08, left=0., right=0.65)
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(GridSpec(1, 1, left=0.0, right=0.45, top=0.95, bottom=0.05)[:, :], projection='3d')
    gs = GridSpec(3, 2, left=0.55, top=0.95, bottom=0.05)

    # build figure from bottom to top
    lList = [
        ('STIM', layerdict_stim),
        ('EX', layerdict_ex),
        ('IN', layerdict_in),
        ]
    pops_list = ['IN', 'EX', 'STIM'] # bottom, center, top
    dots = []
    for pop in pops_list: # from bottom to top
        plot_layer(ax, pop, pops_list)
        dots = plot_connections(ax, pop, pops_list, red_conn_dens, dots)
    plot_dots(ax, dots)

    # make plot look nice
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)', labelpad=-1)
    ax.set_ylabel('y (mm)', labelpad=-1)
    ax.set_xticks([-2., -1, 0, 1., 2.])
    ax.set_yticks([-2., -1, 0, 1., 2.])
    ax.xaxis.set_tick_params(pad=-1)
    ax.yaxis.set_tick_params(pad=-1)
    ax.w_zaxis.line.set_lw(0.)
    ax.set_zticks([])

    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # legend
    handles = \
        [Patch(color=pops['STIM']['color']),
         Patch(color=pops['EX']['color']),
         Patch(color=pops['IN']['color']),
         plt.Line2D((0,1),(0,0), color='white', marker='o',
                    markeredgecolor='black', linestyle=''),
         plt.Line2D((0,1),(0,0), color=pops['EX']['conn_color']),
         plt.Line2D((0,1),(0,0), color=pops['IN']['conn_color'])]
    labels = \
        ['STIM',
         'EX',
         'IN',
         'source',
         'exc. connection',
         'inh. connection']
    ax.legend(handles, labels, numpoints=1, loc=2, bbox_to_anchor=(0.7, 0.95),
              fontsize=10)

    ax.view_init(elev=12, azim=-60)

    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.text(0.05, 0.95, 'A',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='demibold',
        transform=fig.transFigure)

    # plot connectivity using ConnPlotter's style
    conns = [[1, 0], [1, 1], [1, 1]]
    pList = ['EX', 'IN']
    cDicts = [conn_dict_stim, conn_dict_ex, conn_dict_in]
    for i, ((pre, lDict), cDict, conn) in enumerate(zip(lList, cDicts, conns)):
        for j, post in enumerate(pList):
            ax = fig.add_subplot(gs[i, j], aspect='equal')
            extent = lDict['extent']
            x = np.linspace(-extent[0]/2, extent[0]/2, 101)
            y = np.linspace(-extent[1]/2, extent[1]/2, 101)
            X,Y = np.meshgrid(x, y)
            C = np.zeros(X.shape)
            if conn[j]:
                if 'kernel' not in cDict.keys() or cDict['kernel'] == 1.:
                    try:
                        weights = epsilon_stim = num_stim_conn*N_stim / NE * cDict['weights']
                        C[np.sqrt(X**2 + Y**2) <= cDict['mask']['circular']['radius']] = weights
                        # cmap = 'gray_r'
                        colors = [(1, 1, 1), (1, 0, 0)]
                        cmap =  LinearSegmentedColormap.from_list('reds', colors, N=64)
                        vmin = 0
                        vmax = weights
                    except KeyError as ae:
                        raise ae
                if type(cDict['kernel']) is dict:
                    try:
                        sigma = cDict['kernel']['gaussian']['sigma']
                        if 'mask' in cDict.keys():
                            mask = np.sqrt(X**2 + Y**2) <= cDict['mask']['circular']['radius']
                            weights = cDict['weights']*epsilon
                            C[mask] = weights*np.exp(-(X[mask]**2 + Y[mask]**2) / (2*sigma**2)) # / (2*np.pi*sigma**2)
                            if weights > 0:
                                colors = [(1, 1, 1), (1, 0, 0)]
                                cmap =  LinearSegmentedColormap.from_list('reds', colors, N=64)
                                vmin = 0
                                vmax = weights
                            else:
                                colors = [(0, 0, 1), (1, 1, 1)]
                                cmap =  LinearSegmentedColormap.from_list('blues', colors, N=64)
                                vmin = weights
                                vmax = 0
                        else:
                            weights = cDict['weights']
                            C = weights*np.exp(-(X**2 + Y**2) / (2*sigma**2)) # / (2*np.pi*sigma**2)
                            if weights > 0:
                                colors = [(1, 1, 1), (1, 0, 0)]
                                cmap =  LinearSegmentedColormap.from_list('reds', colors, N=64)
                                vmin = 0
                                vmax = weights
                            else:
                                colors = [(0, 0, 1), (1, 1, 1)]
                                cmap =  LinearSegmentedColormap.from_list('blues', colors, N=64)
                                vmin = weights
                                vmax = 0
                    except KeyError as ae:
                        raise ae
                else:
                    pass
            im = ax.pcolormesh(X,Y,C, cmap=cmap, vmin=vmin, vmax=vmax)

            if j == (len(pList)-1):
                bbox = np.array(ax.get_position())
                cax = fig.add_axes([bbox[1][0]+0.01, bbox[0][1], 0.015, (bbox[1][1]-bbox[0][1])])
                axcb = fig.colorbar(im, cax=cax, orientation='vertical')
                cbarlabel = r'$\epsilon_\mathrm{Y,%s}Jg_\mathrm{Y,%s}$ (pA)' % (pre,pre)
                axcb.set_label(cbarlabel)

            ax.set_xticks([-2., -1, 0, 1., 2.])
            ax.set_yticks([-2., -1, 0, 1., 2.])

            if i != (len(lList)-1):
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('x (mm)', labelpad=0)
            if i == 0:
                ax.set_title(post)
            if j != 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel('{}\ny (mm)'.format(pre), labelpad=0)

            if i == 0 and j == 0:
                ax.text(0.5, 0.95, 'B',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=16, fontweight='demibold',
                    transform=fig.transFigure)


    fig.savefig(os.path.join(spike_output_path, 'network_sketch.pdf'), dpi=160,
                bbox_inches='tight')

'''
Definition of helper functions for the network sketch.
'''

def plot_layer(ax, pop, pops_list):
    # plot neurons at their original location
    pos = np.array(pops[pop]['pos']).transpose()
    z0 = pops_list.index(pop)
    ax.plot(pos[0], pos[1], zs=z0, marker='o', markeredgecolor='none',
            linestyle='none', markersize=1, color=pops[pop]['color'],
            alpha=1.)
    ax.text(-2, -2.8, z0+0.3, pop)
    return

def plot_connections(ax, pop, pops_list, red_conn_dens, dots):
    # note that xyloc of connection is set here manually

    # connections from pop to tagt
    for tgt in pops[pop]['tgts']:
        z0 = pops_list.index(pop)
        z1 = z0 + (pops_list.index(tgt) - z0)
        if pop == tgt or z0 <= z1:
            if pop == tgt:
                xyloc = [0.8, 0.8]
            else:
                xyloc = [-0.8, -0.8]
            srcid = tp.FindNearestElement(pops[pop]['layer'], xyloc, False)
            srcloc = tp.GetPosition(srcid)[0]
            tgtsloc = np.array(tp.GetTargetPositions(srcid,
                                                     pops[tgt]['layer'])[0])
            # targets do not get picked in the same order;
            # they are sorted here for reproducibility
            tgtsloc = tgtsloc[np.argsort(tgtsloc[:,0])]
            tgtsloc_show = tgtsloc[0:len(tgtsloc):red_conn_dens]
            for tgtloc in tgtsloc_show:
                plt.plot([srcloc[0], tgtloc[0]], [srcloc[1], tgtloc[1]],
                         [z0, z1], c=pops[pop]['conn_color'], linewidth=1,
                         alpha=0.1)
                # highlight target
                plt.plot(tgtloc[0], tgtloc[1], zs=[z1], marker='o',
                         markeredgecolor='none',
                         markersize=2, color=pops[pop]['conn_color'],
                         alpha=1.)
            dots.append([srcloc, z0, 'white', 'black', 3])
            if pop == 'IN' and tgt == 'EX': # final
                dots.append([tgtsloc_show, z1, pops[pop]['conn_color'],
                             'none', 2])

    # draw connections from src to pop
    for src in pops_list:
        z0 = pops_list.index(src)
        z1 = z0 + (pops_list.index(pop) - z0)
        if src != pop and (pop in pops[src]['tgts']) and z0 > z1:
            if src == 'STIM':
                xyloc = [0.,0.]
            else:
                xyloc = [0.8, -0.8]
            srcid = tp.FindNearestElement(pops[src]['layer'], xyloc, False)
            srcloc = tp.GetPosition(srcid)[0]
            tgtsloc = np.array(tp.GetTargetPositions(srcid,
                                                     pops[pop]['layer'])[0])
            tgtsloc = tgtsloc[np.argsort(tgtsloc[:,0])]
            tgtsloc_show = tgtsloc[0:len(tgtsloc):red_conn_dens]
            for tgtloc in tgtsloc_show:
                plt.plot([srcloc[0], tgtloc[0]], [srcloc[1], tgtloc[1]],
                         [z0, z1], c=pops[src]['conn_color'], linewidth=1,
                         alpha=0.1)
                plt.plot(tgtloc[0], tgtloc[1], zs=[z1], marker='o',
                         markeredgecolor='none',
                         markersize=2, color=pops[src]['conn_color'],
                         alpha=1.)
            dots.append([srcloc, z0, 'white', 'black', 3])
            if src == 'STIM': # final
                dots.append([tgtsloc_show, z1, pops[src]['conn_color'],
                             'none', 2])
    return dots


def plot_dots(ax, dots):
    for i,data in enumerate(dots):
        if type(data[0][0]) == np.ndarray:
            xs = zip(*data[0])[0]
            ys = zip(*data[0])[1]
        else:
            xs = [data[0][0]]
            ys = [data[0][1]]
        ax.plot(xs, ys, zs=data[1], marker='o', markeredgecolor=data[3],
                markersize=data[4], c=data[2], linestyle='none', alpha=1.)
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


    fig = plt.figure(figsize=(8., 8.))
    fig.subplots_adjust(top=0.94, bottom=0.06, left=0.11, right=0.96,
                        wspace=0.3, hspace=1.)
    gs = gridspec.GridSpec(6,5)


    # A: unsorted raster
    gs_cell = gs[:2, :4]
    pops_list = ['STIM', 'EX', 'IN'] # top to bottom
    _plot_raster_unsorted('A', gs_cell, pops_list, times, dilute)


    # B: spike count histogram over unit
    gs_cell = gs[:2, 4]
    pops_list = ['STIM', 'EX', 'IN']
    _plot_unit_histogram('B', gs_cell, pops_list)


    # C: sorted raster
    gs_cell = gs[2:4, :4]
    pops_list = ['EX', 'IN', 'STIM']
    _plot_raster_sorted('A', gs_cell, pops_list, times, dilute)


    # D: spike count histogram over space
    gs_cell = gs[2:4, 4]
    pops_list = ['IN', 'EX', 'STIM'] # bottom to top
    _plot_space_histogram('D', gs_cell, pops_list)


    # E: spike count histogram over time
    gs_cell = gs[4:6, :4]
    pops_list = ['STIM', 'EX', 'IN'] # top to bottom
    _plot_time_histogram('E', gs_cell, pops_list, times)


    # legend to the bottom right
    ax = plt.subplot(gs[4:6,4:]) # just for the legend
    plt.axis('off')
    handles = [Patch(color=pops['STIM']['color']),
               Patch(color=pops['EX']['color']),
               Patch(color=pops['IN']['color'])]
    labels = ['STIM',
              'EX',
              'IN']
    ax.legend(handles, labels, loc='center')

    fig.savefig(os.path.join(spike_output_path, 'raster.pdf'), dpi=160)


'''
Definition of helper functions for the spiking activity.
'''

def _plot_spikes(ax, dilute, nodes=pops['EX']['nodes'],
                 events=pops['EX']['events'],
                 layerdict=pops['EX']['layerdict'],
                 color='r',
                 marker='.', poplabel='EX',
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

    ax.plot(T[::dilute], X[::dilute], marker, markersize=1., color=color, label=poplabel,
                rasterized=True)
    return


def _plot_space_histogram(label, gs_cell, pops_list):
    gs_loc = gridspec.GridSpecFromSubplotSpec(1,3, gs_cell, wspace=0.15)

    binsize=0.05
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
        ax.hist(data[pop], bins=bins, histtype='step',
                color=pops[pop]['color'], orientation='horizontal')
        ax.set_ylim(bins[0], bins[-1])
        ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2, prune='upper'))
        plt.xticks(rotation=-90)

        if i==0:
            ax.text(-0.6, 1.05, label, ha='left', va='bottom', fontsize=16,
                    fontweight='demibold', transform=ax.transAxes)
        if i==len(pops_list)/2:
            ax.set_title('spike count')
            ax.set_xlabel('count')
    return


def _plot_time_histogram(label, gs_cell, pops_list, times):
    gs_loc = gridspec.GridSpecFromSubplotSpec(3,1, gs_cell, hspace=0.15)
    bins = np.arange(transient, simtime+1, 1) # 1 ms bins
    for i,pop in enumerate(pops_list):
        ax = plt.subplot(gs_loc[i,0])
        ax.hist(pops[pop]['events']['times'], bins=bins, histtype='step',
                color=pops[pop]['color'])
        ax.set_ylim(bottom=0) # fixing only the bottom
        ax.set_xlim(times[0], times[1])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='upper'))

        if i==0:
            ax.set_title('spike count')
            ax.text(-0.05, 1.05, label, ha='left', va='bottom', fontsize=16,
                    fontweight='demibold', transform=ax.transAxes)
        if i==len(pops_list)/2:
            ax.set_ylabel('count')
        if i==len(pops_list)-1:
            ax.set_xlabel('time (ms)')
        else:
            ax.set_xticklabels([])
    return


def _plot_unit_histogram(label, gs_cell, pops_list):
    tot_neurons = 0
    ratios = []
    for pop in pops_list:
        tot_neurons += pops[pop]['N']
    for pop in pops_list:
        frac = 1.*pops[pop]['N']/tot_neurons
        if frac < 0.1:
            frac = 0.1
        ratios.append(frac)

    binsize = 20. # neurons

    # maximum estimated spike count, used for setting xlim for all populations
    max_estim_cnt = 0
    for i,pop in enumerate(pops_list):
        estim_cnt = pops[pop]['rate'] * (simtime-transient)*1e-3 * binsize
        if estim_cnt >= max_estim_cnt:
            max_estim_cnt = estim_cnt

    gs_loc = gridspec.GridSpecFromSubplotSpec(3, 1, gs_cell,
                                              height_ratios=ratios,
                                              hspace=0.1)
    for i,pop in enumerate(pops_list):
        ax = plt.subplot(gs_loc[i, 0])

        bins = np.arange(np.min(pops[pop]['nodes']),
                         np.max(pops[pop]['nodes'])+binsize, binsize)
        ax.hist(pops[pop]['events']['senders'],
                bins=bins, histtype='step',
                color=pops[pop]['color'],
                orientation='horizontal', stacked=False, alpha=1)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
        ax.set_xlim(0, 1.3*max_estim_cnt)
        ax.set_xlabel('')
        ax.set_ylim(np.min(pops[pop]['nodes']), np.max(pops[pop]['nodes']))
        ax.set_yticks([])
        ax.set_yticklabels([])

        if i==0:
            ax.text(-0.18, 1.7, label, ha='left', fontsize=16, va='bottom',
                    fontweight='demibold', transform=ax.transAxes)
            ax.set_title('spike count')
        if i!=2:
            #ax.set_xticks([0, ax.axis()[1]])
            ax.set_xticklabels([])
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

    gs_loc = gridspec.GridSpecFromSubplotSpec(3, 1, gs_cell,
                                              height_ratios=ratios,
                                              hspace=0.1)
    for i,pop in enumerate(pops_list):
        ax = plt.subplot(gs_loc[i,0])
        _plot_spikes(ax, dilute, nodes=pops[pop]['nodes'],
                     events=pops[pop]['events'],
                     layerdict=pops[pop]['layerdict'],
                     color=pops[pop]['color'],
                     marker='.', poplabel=pop,
                     position_sorted=False)
        ax.set_xlim(times[0], times[1])
        ax.set_xticklabels([])
        ax.set_xlabel('')
        ax.set_ylim(np.min(pops[pop]['nodes']), np.max(pops[pop]['nodes']))
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel(pops[pop]['N']) # population size
        if i==0:
            ax.set_title('unsorted spike raster')
            ax.text(-0.05, 1.7, label, fontsize=16, ha='left',
                    va='bottom', fontweight='demibold', transform=ax.transAxes)
        if i==1:
            ax.set_ylabel('neuron id\n' + str(pops[pop]['N']))
    return


def _plot_raster_sorted(label, gs_cell, pops_list, times, dilute):
    ax = plt.subplot(gs_cell)
    for pop in pops_list:
        _plot_spikes(ax, dilute, nodes=pops[pop]['nodes'],
                     events=pops[pop]['events'],
                     layerdict=pops[pop]['layerdict'],
                     color=pops[pop]['color'],
                     marker='.', poplabel=pop,
                     position_sorted=True)

    ax.set_title('sorted_spike_raster')
    ax.set_ylabel('x position (mm)')
    ax.set_xlim(times[0], times[1])
    ax.set_xticklabels([])
    ax.text(-0.05, 1.05, 'C', ha='left', fontsize=16, va='bottom',
            fontweight='demibold', transform=ax.transAxes)
    return



if __name__=='__main__':
    # these functions are needed for generating test data for VIOLA
    merge_spike_files()
    write_population_GIDs()
    create_viola_config_raw()

    # these functions are optional
    figure_network_sketch()

    times = [transient, simtime] # displayed time interval
    #times = [simtime - 500., simtime]
    figure_raster(times)
