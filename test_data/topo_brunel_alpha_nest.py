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
using the NEST topology module.

The script writes to the output folder 'out_raw':
- neuron positions
- population GIDs
- plot of spike raster

- raw spike data in .gdf format for VIOLA
- configuration file (for raw data) for VIOLA

Usage:
::

    python topo_brunel_alpha_nest.py
'''

'''
Importing all necessary modules for simulation, analysis and plotting.
'''

from scipy.optimize import fsolve

import nest
import nest.raster_plot
import nest.topology as tp

import time
import os
import glob
import numpy as np
from numpy import exp, random, zeros_like, r_


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

dt      = 0.1    # the resolution in ms
simtime = 1000. # Simulation time in ms

'''
Definition of the parameters crucial for asynchronous irregular firing
of the neurons.
'''

g       = 5.0  # ratio inhibitory weight/excitatory weight
eta     = 2.0  # external rate relative to threshold rate
epsilon = 0.1  # connection probability

'''
Definition of the number of neurons in the network and the number of
neuron recorded from.
'''

order     = 2500
NE        = 4*order # number of excitatory neurons
NI        = 1*order # number of inhibitory neurons
N_neurons = NE+NI   # number of neurons in total

'''
Definition of connectivity parameters
'''

CE    = int(epsilon*NE) # number of excitatory synapses per neuron
CI    = int(epsilon*NI) # number of inhibitory synapses per neuron  
C_tot = int(CI+CE)      # total number of synapses per neuron

'''
Initialization of the parameters of the integrate-and-fire neurons and
the synapses. The parameter of the neuron are stored in a dictionary.
The synaptic currents are normalized such that the amplitude of the
PSP is J.
'''

tauSyn = 0.5  # synaptic time constant in ms
tauMem = 20.0 # time constant of membrane potential in ms
CMem = 250.0  # capacitance of membrane in in pF
theta  = 20.0 # membrane threshold potential in mV
neuron_params= {"C_m":        CMem,
                "tau_m":      tauMem,
                "tau_syn_ex": tauSyn,
                "tau_syn_in": tauSyn,
                "t_ref":      2.0,
                "E_L":        0.0,
                "V_reset":    0.0,
                "V_m":        0.0,
                "V_th":       theta}
J      = 0.1        # postsynaptic amplitude in mV
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
p_rate = 1000.0*nu_ex*CE

'''
Definition of topology-specific parameters. Connection routines use fixed
indegrees = convergent connections with a fixed number of connections.
'''

extent_length = 4.   # in mm (layer size = extent_length x extent_length)
mask_radius = 2.     # mask radius in mm
sigma = 0.3          # Gaussian profile, sigma in mm

layerdict_EX = {
    'extent' : [extent_length, extent_length],
    'positions' : [[(random.rand()-0.5)*extent_length,
                    (random.rand()-0.5)*extent_length] for x in range(NE)],
    'elements' : 'iaf_psc_alpha',
    'edge_wrap' : True, # PBC
}

layerdict_IN = {
    'extent' : [extent_length, extent_length],
    'positions' : [[(random.rand()-0.5)*extent_length,
                    (random.rand()-0.5)*extent_length] for x in range(NI)],
    'elements' : 'iaf_psc_alpha',
    'edge_wrap' : True,
}

conn_dict_EX = {
    'connection_type': 'convergent',
    'allow_autapses': False,
    'allow_multapses': True,
    'weights' : J_ex,
    'delays' : {
        'linear' : {
            'c' : 1.5,
            'a' : 0.,
            }
        },
    'mask' : {
        'circular' : {'radius' : mask_radius}
        },
    'kernel' : {
        'gaussian' : {
            'p_center' : 1.,
            'sigma' : sigma,
            'mean' : 0.,
            'c' : 0.,
            }
        },
    'number_of_connections' : CE,
    }

conn_dict_IN = {
    'connection_type': 'convergent',
    'allow_autapses': False,
    'allow_multapses': True,
    'weights' : J_in,
    'delays' : {
        'linear' : {
            'c' : 1.5,
            'a' : 0.,
            }
        },
    'mask' : {
        'circular' : {'radius' : mask_radius}
        },
    'kernel' : {
        'gaussian' : {
            'p_center' : 1.,
            'sigma' : sigma,
            'mean' : 0.,
            'c' : 0.,
            }
        },
    'number_of_connections' : CI,
    }


'''
Destination for spike output and definition of file prefixes.
'''

spike_output_path = 'out_raw'
label = 'spikes' # spike detectors
label_positions = 'neuron_positions' # neuron positions

'''
Create the file output destination folder if it does not exist.
'''

if not os.path.isdir(spike_output_path):
    os.mkdir(spike_output_path)

'''
Reset the simulation kernel.
Configuration of the simulation kernel by the previously defined time
resolution used in the simulation. Setting "print_time" to True prints
the already processed simulation time as well as its percentage of the
total simulation time.
'''

nest.ResetKernel()
nest.SetKernelStatus({"resolution": dt,
                      "print_time": True,
                      "overwrite_files": True,
                      'total_num_virtual_procs': 4})

print("Building network")

'''
Configuration of the model `iaf_psc_alpha` and `poisson_generator`
using SetDefaults(). This function expects the model to be the
inserted as a string and the parameter to be specified in a
dictionary. All instances of theses models created after this point
will have the properties specified in the dictionary by default.
'''

nest.SetDefaults("iaf_psc_alpha", neuron_params)
nest.SetDefaults("poisson_generator",{"rate": p_rate})

'''
Creation of the topology layers for excitatory and inhibitory neurons.
GIDs and neuron positions are written to file.
'''

layer_ex = tp.CreateLayer(layerdict_EX)
layer_in = tp.CreateLayer(layerdict_IN)

tp.DumpLayerNodes(layer_ex, os.path.join(spike_output_path,
                                         label_positions + '-0.dat'))
tp.DumpLayerNodes(layer_in, os.path.join(spike_output_path,
                                         label_positions + '-1.dat'))

nodes_ex = nest.GetChildren(layer_ex)[0] # nodes of ex/in neurons
nodes_in = nest.GetChildren(layer_in)[0]

'''
Create spike detectors for recording from the excitatory and the
inhibitory populations and a poisson generator as noise source.
The spike detectors are configured for writing to file.
'''

espikes=nest.Create("spike_detector")
ispikes=nest.Create("spike_detector")

nest.SetStatus(espikes,[{
                   "label": os.path.join(spike_output_path, label + "-0"),
                   "withtime": True,
                   "withgid": True,
                   "to_file": True,
                   }])

nest.SetStatus(ispikes,[{
                   "label": os.path.join(spike_output_path, label + "-1"),
                   "withtime": True,
                   "withgid": True,
                   "to_file": True,
                   }])

noise=nest.Create("poisson_generator")

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
Connecting the excitatory and inhibitory populations to the associated spike
detectors using excitatory synapses. Here the same shortcut for the
specification of the synapse as defined above is used.
'''

nest.Connect(nodes_ex, espikes, syn_spec="excitatory")
nest.Connect(nodes_in, ispikes, syn_spec="excitatory")

print("Connecting network")

'''
Connecting the excitatory and inhibitory populations using the
pre-defined excitatory/inhibitory synapse and the connection dictionaries.
First, update the connection dictionaries with the synapses.
'''

conn_dict_EX['synapse_model'] = 'excitatory'
conn_dict_IN['synapse_model'] = 'inhibitory'

print("Excitatory connections")

tp.ConnectLayers(layer_ex, layer_ex, conn_dict_EX)
tp.ConnectLayers(layer_ex, layer_in, conn_dict_EX)

print("Inhibitory connections")
tp.ConnectLayers(layer_in, layer_ex, conn_dict_IN)
tp.ConnectLayers(layer_in, layer_in, conn_dict_IN)

'''
Storage of the time point after the buildup of the network in a
variable.
'''

endbuild=time.time()

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

'''
Calculation of the average firing rate of the excitatory and the
inhibitory neurons by the simulation time. The
multiplication by 1000.0 converts the unit 1/ms to 1/s=Hz.
'''

rate_ex   = events_ex/simtime*1000./len(nodes_ex)
rate_in   = events_in/simtime*1000./len(nodes_in)

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
print("Building time     : %.2f s" % build_time)
print("Simulation time   : %.2f s" % sim_time)


'''
Merging spike files, Writing population GIDs and a configuration
file for viola to file and plotting a spike raster.
'''

def merge_spike_files():
    '''
    merge spike files from different threads
    '''
    for i, pop in enumerate(['EX', 'IN']):
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
    write first and last neuron GID of both poulations to file
    '''
    fname = os.path.join(spike_output_path, 'population_GIDs.dat')
    with open(fname, 'w') as f:
        f.write('%d\t%d\n' % (nodes_ex[0], nodes_ex[-1]))
        f.write('%d\t%d\n' % (nodes_in[0], nodes_in[-1]))
    f.close()

merge_spike_files()
write_population_GIDs()

import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import json

#population colors
popColors = []
cmap = plt.get_cmap('rainbow_r', 2)
for i in range(cmap.N):
    rgb = cmap(i)[:3]
    col_hex = mpc.rgb2hex(rgb)
    popColors.append(col_hex)
popColors = ','.join(popColors)

config_dict = {}
config_dict.update({
    "popNum": 2,
    "popNames": ','.join(['EX','IN']),
    "spikesFiles": [label+'-%i.gdf' % X for X in [0,1]],
    "timestamps": int(simtime / dt),
    "resolution": dt,
    "xSize": extent_length,
    "ySize": extent_length,
    "dataType": "neuron",
    "posFiles": [label_positions+'-%i.dat' % X for X in [0,1]],
    "xLFP": 10,
    "yLFP": 10,
    "timelineLenght": 40,
    "popColors": popColors,
})

with open(os.path.join(spike_output_path, 'config_raw.json'), 'w') as f:
    json.dump(config_dict, f)


'''
Plotting.
'''

# rasters and histograms from nest
if False:
    eraster = nest.raster_plot.from_device(espikes, hist=True)
    iraster = nest.raster_plot.from_device(ispikes, hist=True)

#sorted raster plot:
if True:
    import matplotlib.pyplot as plt

    # stepsize for diluting (1 = all)
    dilute = int(5) # int

    eevents = nest.GetStatus(espikes, 'events')[0]
    ievents = nest.GetStatus(ispikes, 'events')[0]

    def plot_spikes(nodes=nodes_ex, events=eevents,
                    layerdict=layerdict_EX,
                    color='r',
                    marker='.', poplabel='EX'):
        '''
        plot sorted spike raster, flexible for both populations
        '''
        X = []
        T = []
        for i, j in enumerate(nodes):
            #extract spikes
            t = events['times'][events['senders'] == j]
            x, y = layerdict['positions'][i]
            if t.size > 0:
                T = r_[T, t] # concatenate spike times
                X = r_[X, zeros_like(t)+x] # sorted by x positions
        # dilute
        X = X[np.arange(0, len(X), dilute)]
        T = T[np.arange(0, len(T), dilute)]
        ax.plot(T, X, marker, color=color, label=poplabel, rasterized=True)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    plot_spikes(nodes=nodes_ex, events=eevents,
                layerdict=layerdict_EX,
                color=cmap(0),
                marker='.', poplabel='EX')
    plot_spikes(nodes=nodes_in, events=ievents,
                layerdict=layerdict_IN,
                color=cmap(1),
                marker='.', poplabel='IN')

    ax.axis()
    ax.legend(loc=1)
    ax.set_title('sorted spike raster')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('x position (mm)')

    fig.savefig(os.path.join(spike_output_path, 'sorted_raster.pdf'))
    plt.show()
