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


from nest_preprocessing import *

print('perform spatiotemporal binning of network activity for LFPs')

#input and output path must be provided on the command line, or in
#ipython as additional args when calling run, e.g.,
#>>>run nest_preprocessing.py output_raw output_processed
if len(sys.argv) != 3:
    raise Exception('please provide input path and output path')

input_path = sys.argv[-2]
output_path = sys.argv[-1]

#bin spike data on a grid with a spatial bin size .4 mm and temporal bin size
#of 1 ms
preprocess = ViolaPreprocessing(input_path=input_path,
                            output_path=output_path,
                            X = ['EX', 'IN'],
                            t_sim = 1000.,
                            dt = 0.1,
                            extent_length = 4.,
                            GID_filename = 'population_GIDs.dat',
                            position_filename_label = 'neuron_positions-',
                            spike_detector_label = 'spikes-',
                            TRANSIENT=0.,
                            BINSIZE_TIME=1.,
                            BINSIZE_AREA=0.4,
)


#some needed attributes
preprocess.GIDs_corrected = preprocess.get_GIDs()
positions_corrected = preprocess.get_positions()
#bin-coordinates in the spatially resolved spike histograms
y, x = np.meshgrid(preprocess.pos_bins[:-1], preprocess.pos_bins[:-1])
y = y.flatten() + .2
x = x.flatten() + .2


#make some mockup spike-LFP kernels for E-E, E-I, I-E, I-I connections in
#bin-bin distances up to half along the diagonal.
r = np.unique(np.sqrt((x-x[0])**2 + (y-y[0])**2))
r = r[r < np.sqrt(2*(preprocess.extent_length/2.)**2)]

#mockup spike-LFP relationship as alpha-function with amplitude
#decaying as (1/(1+r**-2)), and with time constant increasing slightly with
#distance. Note, that no heterogeneity in kernel shape except sign is introduced
#for excitatory and inhibitory connections and population output.
lag = 25
t = np.arange(lag)
tau = 2. + t/10.
h = np.zeros((r.size, lag*2 + 1))
for i, a in enumerate(.1/(.1+r**2)):
    alpha = np.r_[np.zeros(lag+1), t*np.exp(-t/tau[i])]
    alpha /= alpha.max()
    h[i, ] = a*alpha
h /= 10. #scale to more realistic "LFP" magnitudes in (mV)


print('reconstructing LFP from kernels....')

#combine weight-modification factors
weight_mod_facts = np.array([[1, -5], [1, -5]]) # 1 and -g of network

#container for reconstructed LFP per postsynaptic population
LFP_h = {}
#iterate over presynaptic populations, then postsynaptic populations
for i, X in enumerate(preprocess.X):
    print('presynaptic population {}'.format(X))
    #compute spike train histograms for each 0.4x0.4 mm bin centered on each
    #contact using same procedure as in dataset_analysis.py
    spikes = read_gdf(os.path.join(preprocess.output_path,
                                        preprocess.spike_detector_label +
                                        X + '.gdf'))
    sptrains = preprocess.compute_time_binned_sptrains(X, spikes,
                                                       preprocess.time_bins_rs,
                                                       dtype=np.uint8)
    binned_sptrains = preprocess.compute_pos_binned_sptrains(positions_corrected[X],
                                                           sptrains,
                                                           dtype=np.uint16).toarray()

    for j, Y in enumerate(preprocess.X):
        #Set up container for LFP signal of each postsynaptic population
        #due to presynaptic activity
        if Y not in LFP_h.keys():
            LFP_h[Y] = np.zeros(binned_sptrains.shape)
        #as kernels are computed using unit current amplitudes, the kernel LFP
        #has to be scaled accounting for network synapse strengths and sign
        w = weight_mod_facts[j, i]

        #np.convolve can only deal with 1D sequences, so we have to recursively
        #iterate over all local and non-local rate bins.
        #
        #Convolve rate bin with corresponding kernel:
        for k in range(x.size):
            #iterate over distances.
            for l, d in enumerate(r):
                #compute rate-bin distance to other bins, taking into account
                #periodic boundary conditions
                if True: #We're using periodic boundary conditions
                    xdist = np.abs(x-x[k])
                    ydist = np.abs(y-y[k])
                    xdist[xdist > preprocess.extent_length] = preprocess.extent_length - xdist[xdist > preprocess.extent_length]
                    ydist[ydist > preprocess.extent_length] = preprocess.extent_length - ydist[ydist > preprocess.extent_length]
                    R = np.sqrt(xdist**2 + ydist**2)
                else:
                    R = (np.sqrt((x-x[k])**2 + (y-y[k])**2))

                inds = R == d
                if inds.sum() > 0:
                    #Convolve, add contribution to fake LFP signal
                    LFP_h[Y][inds, ] += np.convolve(binned_sptrains[k, ],
                                                 h[l, ], 'same')*w

#write the cell-type specific output files to hdf5.
#Compute the compound signal as well
if False:
    LFP_approx = np.zeros_like(LFP_h[Y])
    for key, value in LFP_h.items():
        LFP_approx += value
        f = h5py.File(os.path.join(preprocess.output_path, key + 'LFP.h5'), 'w')
        f['data'] = value
        f['srate'] = 1. / preprocess.BINSIZE_TIME
        print 'wrote file {}'.format(f)
        f.close()

    #write compound output file
    f = h5py.File(os.path.join(preprocess.output_path, 'LFP.h5'), 'w')
    f['data'] = LFP_approx
    f['srate'] = 1. / preprocess.BINSIZE_TIME
    print 'wrote file {}'.format(f)
    f.close()

    #dump data as text file for use with the Visualizer.
    os.system('h5dump -d data -o {}/LFPData.lfp -w 0 -y {}/LFP.h5'.format(
        preprocess.output_path, preprocess.output_path))
else:
    #save directly on text-based file format the visualizer can read,
    #as e.g., hdf5 is not supported. First the population specific signals, that
    #will be identical by design, then, the compound signal
    LFP_approx = np.zeros_like(LFP_h[Y])
    for key, value in LFP_h.items():
        LFP_approx += value
        np.savetxt(os.path.join(preprocess.output_path, key + 'LFPdata.lfp'),
                   value, fmt='%.4f', delimiter=', ')

    #compound signal
    np.savetxt(os.path.join(preprocess.output_path, 'LFPdata.lfp'), LFP_approx,
               fmt='%.4f', delimiter=', ')


print('done!')
