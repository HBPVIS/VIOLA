#!/usr/bin/env python

'''
Generation of LFP like signal from spike data
---------------------------------------------

Generates "LFP" like signal for demonstration with the VIOLA tool, not to be
confused with a realistic LFP signal in the sense that it is computed from
transmembrane currents. Here, we assume a stereotypical spike-LFP relationship
that is identical for both excitatory and inhibitory connections and populations
sans sign.

The setup is such that (i) all spikes from the network is assigned to bins of
0.4 x 0.4 mm in a 10 x 10 layout covering the network, with a time resolution
of 1 ms. The resulting binwise rate profiles are convolved with spatiotemporal
kernels that could in principle be representative of the spike-lfp relation
when network correlations are ignored.


The main output is the file out_proc/LFPdata.lfp containing the
signal in 100 channels across the network plane.

Usage:
::

    python fake_LFP_signal.py out_raw out_proc
'''
from __future__ import division
import sys
import os
import numpy as np
import scipy.signal as ss
import nest_preprocessing as npr
import topo_brunel_alpha_nest as network
import LFPy
import neuron
import quantities as pq
import matplotlib.pyplot as plt


def decimate(x, q=10, n=4, k=0.8, filterfun=ss.cheby1):
    """
    scipy.signal.decimate like downsampling using filtfilt instead of lfilter,
    and filter coeffs from butterworth or chebyshev type 1.


    Parameters
    ----------
    x : numpy.ndarray
        Array to be downsampled along last axis.
    q : int 
        Downsampling factor.
    n : int
        Filter order.
    k : float
        Aliasing filter critical frequency Wn will be set as Wn=k/q.
    filterfun : function
        `scipy.signal.filter_design.cheby1` or
        `scipy.signal.filter_design.butter` function

    Returns
    -------
    numpy.ndarray
        Array of downsampled signal.
              
    """
    if not isinstance(q, int):
        raise TypeError("q must be an integer")

    if n is None:
        n = 1

    if filterfun == ss.butter:
        b, a = filterfun(n, k / q)
    elif filterfun == ss.cheby1:
        b, a = filterfun(n, 0.05, k / q)
    else:
        raise Exception('only ss.butter or ss.cheby1 supported')

    try:
        y = ss.filtfilt(b, a, x)
    except: # Multidim array can only be processed at once for scipy >= 0.9.0
        y = []
        for data in x:
            y.append(ss.filtfilt(b, a, data))
        y = np.array(y)

    try:
        return y[:, ::q]
    except:
        return y[::q]



print('perform spatiotemporal binning of network activity for LFPs')

# input and output path can be provided on the command line:
#    python fake_LFP_signal.py out_raw out_proc
# if no argument is given, default values are used
if len(sys.argv) != 3:
    input_path = 'out_raw'
    output_path = 'out_proc'
else:
    input_path = sys.argv[-2]
    output_path = sys.argv[-1]

#bin spike data on a grid with a spatial bin size .4 mm and temporal bin size
#of 1 ms
preprocess = npr.ViolaPreprocessing(input_path=input_path,
                            output_path=output_path,
                            X = ['EX', 'IN', 'STIM'],
                            t_sim = 2000.,
                            dt = 0.1,
                            extent_length = 4.,
                            GID_filename = 'population_GIDs.dat',
                            position_filename_label = 'neuron_positions-',
                            spike_detector_label = 'spikes-',
                            TRANSIENT=500.,
                            BINSIZE_TIME=network.dt,
                            # BINSIZE_TIME=1.,
                            BINSIZE_AREA=0.4,
)


#some needed attributes
preprocess.GIDs_corrected = preprocess.get_GIDs()
positions_corrected = preprocess.get_positions()
#bin-edge-coordinates in the spatially resolved spike histograms
y, x = np.meshgrid(preprocess.pos_bins[:-1], preprocess.pos_bins[:-1])

# compute for a sentral bin the unique distances to neighbouring and bin
# counts at similar distances.
r, r_counts = np.unique(np.round(np.sqrt(x**2 + y**2), decimals=10), return_counts=True)
inds = r < np.sqrt(2*(preprocess.extent_length/2.)**2)
r = r[inds]
r_counts = r_counts[inds]

# bin senterpoints (flattened)
y, x = np.meshgrid(preprocess.pos_bins[:-1], preprocess.pos_bins[:-1])
x = x.flatten() + .2
y = y.flatten() + .2

# temporal lag vector for the convolution. Temporal bin size of 1 ms is used.
lag = 25

print('precomputing LFP kernels using a ball and stick model')

# derive biophysics params from point-neuron network params (CMem, tauMem)
Ra = 150 # ohm-cm
cm = 1. * pq.uF / pq.cm**2
CMem = network.CMem * pq.pF
r_soma = ((CMem / (4*np.pi*cm))**0.5) # m
# compute g_pas such that membrane time constant of point neuron preserved
tauMem = network.tauMem * pq.ms
g_pas = (cm / tauMem) # s**3*A**2/(kg*m**4) == S / m**2

# convert to dimensionless (NEURON units)
r_soma = (r_soma / pq.um).simplified
g_pas = (g_pas / (pq.S / pq.cm**2)).simplified

cellParams = dict(
    morphology=None,
    Ra=Ra,
    cm=cm,
    passive=True,
    passive_params = dict(g_pas=g_pas, e_pas=-70),
    delete_sections=False,
    tstop=lag*2,
    dt=network.dt    
)

synParams = dict(
    syntype = 'AlphaISyn',
    tau = network.tauSyn,
)
synParams_ex = dict(weight=network.J_ex*1E-3, **synParams)
synParams_in = dict(weight=network.J_in*1E-3, **synParams)

electrodeParams = dict(
    x = r*1E3, # mm -> um
    y = np.zeros(r.size),
    z = np.zeros(r.size),
    sigma = 0.3,
    method='soma_as_point',
    # report the averaged LFP within a square with side length 400 um
    contact_shape='square',
    N = [[0, 0, 1]]*r.size,
    r = 400,
    n = 200,
)

def compute_h(cellParams, synParams, electrodeParams, section='soma'):
    # create ball soma and stick dendrite model
    soma = neuron.h.Section(name='soma')
    soma.diam = r_soma*2
    soma.L = r_soma*2
    
    dend = neuron.h.Section(name='dend')
    dend.diam = 5.
    dend.L = 500.
    
    dend.connect(soma(1.), 0.)
    
    # instantiate LFPy.Cell class
    cell = LFPy.Cell(**cellParams)
    cell.set_pos(0, 0, 0)
    cell.set_rotation(y=-np.pi/2)
        
    # instantiate LFPy.Synapse class
    try:
        assert(hasattr(neuron.h, 'AlphaISyn'))
    except AttributeError:
        raise AttributeError('run nrnivmodl inside this folder')
    
    # create synapses, distribute across entire section so we divide
    # the total synapse input by number of segments in section
    idx = cell.get_idx(section=section)
    weight = synParams.pop('weight')
    for i in idx:
        syn = LFPy.Synapse(cell, idx=i, weight=weight / idx.size,
                           **synParams)
        syn.set_spike_times(np.array([lag]))
    
    # run simulation of extracellular potentials
    cell.simulate(rec_imem=True)
    
    # instantiate RexExtElectrode class and compute the electrode signals    
    electrode = LFPy.RecExtElectrode(cell=cell, **electrodeParams)
    electrode.calc_lfp()
    
    return electrode.LFP

# compute kernels for excitatory and inhibitory connections to the ball and stick
H = dict()
H['EX'] = compute_h(cellParams, synParams_ex, electrodeParams, section='dend')
H['IN'] = compute_h(cellParams, synParams_in, electrodeParams, section='soma')
H['STIM'] = H['EX']

# average out-degrees computed from the fixed indegrees
od_ex = network.N_neurons * network.CE / network.NE
od_in = network.N_neurons * network.CI / network.NI
od_stim = network.num_stim_conn

# modify convolution kernels to account for distance-dependent connection
# probabilities and delays (as well as mean outdegree)

for i, (X, outdegree) in enumerate(zip(preprocess.X, [od_ex, od_in, od_stim])):
    if X == 'EX':
        sigma = network.sigma_ex
    elif X == 'IN':
        sigma = network.sigma_in
    elif X == 'STIM':
        mask_radius_stim = network.mask_radius_stim

    h0 = H[X]
    h1 = np.zeros(h0.shape)
    
    for l, (d, count) in enumerate(zip(r, r_counts)):
        if X == 'STIM':
            delay = int(network.conn_dict_stim['delays'] / network.dt)
        elif X == 'EX':
            c = network.conn_dict_ex['delays']['linear']['c'] # ms
            a = network.conn_dict_ex['delays']['linear']['a'] # mm/ms
            delay = int((c + a*d) / network.dt) # unitless
        elif X == 'IN':
            c = network.conn_dict_in['delays']['linear']['c'] # ms
            a = network.conn_dict_in['delays']['linear']['a'] # mm/ms
            delay = int((c + a*d) / network.dt) # unitless
        else:
            raise Exception
        h_delay = np.zeros(h0.shape[1])
        h_delay[int(lag / network.dt) + delay] = 1.
        
        # distance-dependent connection probability:
        if X != 'STIM':
            p = np.exp(-d**2 / sigma**2) / np.sqrt(2*np.pi*sigma**2)
        else:
            if d < mask_radius_stim:
                p = 1
            else:
                p = 0
        # shift the corresponding LFP kernel in time according to delay
        # and multiply with count of bins at the corresponding distance.
        # We also multiply with the normal function value at this offset
        # to accout for the distance-dependent connection probability.        
        h1[l, ] = np.convolve(h0[l, ], h_delay, 'same')*count*p
        
    # update kernel to modified kernel.
    # The kernel is scaled by the outdegree (# target cells) of the
    # sender population
    H[X] = h1*outdegree

    plt.matshow(h0)
    plt.axis('tight')
    plt.colorbar()
    plt.matshow(h1)
    plt.axis('tight')
    plt.colorbar()
plt.show()

raise Exception


# Container for reconstructed LFP per postsynaptic population
LFP_h = {}
# Iterate over presynaptic populations, then postsynaptic populations.
# Provide the corresponding kernels and scalings by outdegree
for i, X in enumerate(preprocess.X):
    print('presynaptic population {}'.format(X))
    # compute spike train histograms for each 0.4x0.4 mm bin centered on each
    # contact using same procedure as in dataset_analysis.py
    spikes = npr.read_gdf(os.path.join(preprocess.output_path,
                                        preprocess.spike_detector_label +
                                        X + '.gdf'))
    sptrains = preprocess.compute_time_binned_sptrains(X, spikes,
                                                       preprocess.time_bins_rs,
                                                       dtype=np.uint8)
    binned_sptrains = preprocess.compute_pos_binned_sptrains(positions_corrected[X],
                                                           sptrains,
                                                           dtype=np.uint16).toarray()

    # for j, Y in enumerate(preprocess.X[:-1]):
    # Set up container for LFP signal of each postsynaptic population
    # due to presynaptic activity
    if X not in LFP_h.keys():
        LFP_h[X] = np.zeros(binned_sptrains.shape)
        
    # np.convolve can only deal with 1D sequences, so we have to recursively
    # iterate over all local and non-local rate bins.
    for k in range(x.size):
        #iterate over distances.
        for l, d in enumerate(r):
            # compute rate-bin distance to other bins, taking into account
            # periodic boundary conditions
            if True: # We're using periodic boundary conditions
                xdist = np.abs(x-x[k])
                ydist = np.abs(y-y[k])
                xdist[xdist > preprocess.extent_length] = preprocess.extent_length - xdist[xdist > preprocess.extent_length]
                ydist[ydist > preprocess.extent_length] = preprocess.extent_length - ydist[ydist > preprocess.extent_length]
                R = np.round(np.sqrt(xdist**2 + ydist**2), decimals=10)
            else:
                R = np.round((np.sqrt((x-x[k])**2 + (y-y[k])**2)), decimals=10)

            inds = R == d
            if inds.sum() > 0:
                # Convolve, add contribution to compute LFP signal
                LFP_h[X][inds, ] += np.convolve(binned_sptrains[k, ], H[X][l, ], 'same')

# downsample LFP signals to a time resolution of 1 ms. 
for key, value in LFP_h.items():
    LFP_h[key] = decimate(value, q=int(1/network.dt))

# write the cell-type specific output files to hdf5.
# Compute the compound signal as well
if False:
    LFP_approx = np.zeros_like(LFP_h[X])
    for key, value in LFP_h.items():
        LFP_approx += value
        f = h5py.File(os.path.join(preprocess.output_path, key + 'LFP.h5'), 'w')
        f['data'] = value
        f['srate'] = 1., # / preprocess.BINSIZE_TIME
        print 'wrote file {}'.format(f)
        f.close()

    # write compound output file
    f = h5py.File(os.path.join(preprocess.output_path, 'LFP.h5'), 'w')
    f['data'] = LFP_approx
    f['srate'] = 1., # / preprocess.BINSIZE_TIME
    print 'wrote file {}'.format(f)
    f.close()

    #dump data as text file for use with the Visualizer.
    os.system('h5dump -d data -o {}/LFPData.lfp -w 0 -y {}/LFP.h5'.format(
        preprocess.output_path, preprocess.output_path))
else:
    # save directly on text-based file format the visualizer can read,
    # as hdf5 is not supported. First the population specific signals, that
    # will be identical by design, then, the compound signal
    LFP_approx = np.zeros_like(LFP_h[X])
    for key, value in LFP_h.items():
        LFP_approx += value
        np.savetxt(os.path.join(preprocess.output_path, key + 'LFPdata.lfp'),
                   value, fmt='%.4f', delimiter=', ')

    # compound signal
    np.savetxt(os.path.join(preprocess.output_path, 'LFPdata.lfp'), LFP_approx,
               fmt='%.4f', delimiter=', ')


for value in LFP_h.values(): plt.matshow(value); plt.axis('tight'); plt.axis('tight'); plt.colorbar()
plt.show()

print('done!')
