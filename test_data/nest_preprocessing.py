#!/usr/env/python

'''
Preprocessing NEST output for VIOLA
-----------------------------------
This script preprocesses spiking data from a NEST simulation using the
topology module for spatial structure so that it can be loaded to VIOLA as
"binned" data type. Concretely, it creates spatially and temporally binned
spike rates.

Parallelization with MPI is possible.

The script writes to a specified output folder, e.g., 'out_proc':
- spatially binned spike rates in .h5 format
- neuron position data in .h5 format
- neuron population GIDs
- merged spike data in .gdf format

- spatially binned spike rates .dat format for VIOLA
- configuration file (for processed data) for VIOLA

Usage:
::

    python nest_preprocessing.py out_raw out_proc
'''

import sys
import numpy as np
import scipy.sparse as sp
import glob
import h5py
import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
from time import time, sleep
from mpi4py import MPI


###################################
# Initialization of MPI stuff     #
###################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


########################
# Function definitions #
########################


def read_gdf(fname):
    gdf_file = open(fname, 'r')
    gdf = []
    for l in gdf_file:
        data = l.split()
        gdf += [data]
    gdf = np.array(gdf, dtype=object)
    # check data type (float or int) in each column
    # (if mixed, cast to float)
    if gdf.size > 0:
        for col in range(gdf.shape[1]):
            if any ('.' in s for s in gdf[:,col]):
                gdf[:,col] = gdf[:,col].astype(float)
            else:
                gdf[:,col] = gdf[:,col].astype(int)
    return np.array(gdf)


def write_gdf(gdf, fname):
    gdf_file = open(fname,'w')
    for line in gdf:
        for i in np.arange(len(line)):
            gdf_file.write(str(line[i]) + '\t')
        gdf_file.write('\n')
    return None


def dump_sparse_to_h5(X, f, data, compression='gzip', compression_opts=2):
    '''
    write sparse matrix entry to hdf5 file under groupname X

    Arguments
    ---------
    X : str
        top-level group name
    f : file
        <HDF5 file "filename.h5" (mode r+)>
    data : scipy.sparse.coo.coo_matrix
        <NxM sparse matrix>
    compression : str
        compression strategy, see h5py.File.create_dataset
    compression_opts : int
        compression settings, see h5py.File.create_dataset

    '''
    if type(data) == sp.coo_matrix:
        x = data
    else:
        x = data.tocoo()

    group = f.create_group(X)
    dset = group.create_dataset('data_row_col',
                               data=np.c_[x.data, x.row, x.col],
                               compression=compression,
                               compression_opts=compression_opts,
                               maxshape = (None, None))
    dset = group.create_dataset('shape', data=x.shape, maxshape= (None,))


def load_h5_to_sparse(X, f):
    '''load sparse matrix stored on COOrdinate format from HDF5.

    Arguments
    ---------
    X : str
        group name, group must contain datasets:
            'data', 'row', 'col' vectors of equal length
            'shape' : shape of array tuple
    f : file
        <HDF5 file "filename.h5" (mode r+)>

    Returns
    -------
    data : scipy.sparse.csr.csr_matrix

    '''
    data = sp.coo_matrix((f[X]['data_row_col'].value[:, 0],
                          (f[X]['data_row_col'].value[:, 1],
                           f[X]['data_row_col'].value[:, 2])),
                         shape=f[X]['shape'].value)

    return data.tocsr()


##################################
# Class definitions              #
##################################

class ViolaPreprocessing(object):
    '''class VisualizerPreprocessing for processing NEST spike output for
    Vizualizer tool'''
    def __init__(self,
                 input_path,
                 output_path,
                 X = ['EX', 'IN'],
                 t_sim = 1000.,
                 dt = 0.1,
                 extent_length = 4.,
                 GID_filename = 'population_GIDs.dat',
                 position_filename_label = 'neuron_positions-',
                 spike_detector_label = 'spikes-',
                 TRANSIENT=0.,
                 BINSIZE_TIME=1.,
                 BINSIZE_AREA=0.1,
                ):
        '''
        initialization of class ViolaPreprocessing

        Arguments
        ---------
        input_path : path
            path to simulation output
        output_path : str
            path to output folder
        X : list of str
            list of population names, e.g., X=['L23E','L23I',...,'L6I']
        t_sim : float
            simulation duration in ms
        dt : float
            simulation time resolution in ms
        extent_length : float
            side length of square topology layer
        GID_filename : str
            name of file with neuron GIDs for all populations
        position_filename_label : str
            prefix of file(s) containing neuron positions, files matching
            name will be read. Files must have file ending .dat
        spike_detector_label : str
            file prefix of spike output files, such that they can be read by
            the script on the form: spike_detektor_label + '{X}-*.gdf
        TRANSIENT : float
            starting point of preprocess (for removing startup transients),
            transient will be cut off from the beginning of the simulation
            and spike times will start at 0.
        BINSIZE_TIME : float, binsize when temporally binning spikes in ms
        BINSIZE_AREA : float, binsize when spatially binning positions
        '''
        #set class attributes
        self.output_path = output_path
        self.TRANSIENT = TRANSIENT
        self.BINSIZE_TIME = BINSIZE_TIME
        self.BINSIZE_AREA = BINSIZE_AREA


        #time resolution of simulation
        self.dt = dt

        # simulation time
        self.t_sim = t_sim

        # destination of raw nest output
        self.input_path = input_path


        # get extent_length of the topology layer
        # positions are in [-0.5*extent_length, 0.5*extent_length] by default
        self.extent_length = extent_length


        #file name of raw population GIDs
        self.GID_filename = GID_filename


        #file name of GID positions
        self.position_filename_label = position_filename_label


        #file prefix for spike detector
        self.spike_detector_label = spike_detector_label


        #population names
        self.X = X


        # time bins for spike trains start at 0,
        # the simulation time is reduced by the startup transient
        self.time_bins = np.arange((self.t_sim - self.TRANSIENT) \
                                   / self.dt) * self.dt
        self.time_bins_rs = np.arange((self.t_sim - self.TRANSIENT) \
                                      / self.BINSIZE_TIME) * self.BINSIZE_TIME

        #bins for the positions
        self.pos_bins = np.linspace(-self.extent_length / 2,
                                    self.extent_length / 2,
                                    int(self.extent_length /\
                                        self.BINSIZE_AREA + 1))


        # load GIDs
        GID_filename = open(os.path.join(self.input_path,
                                         self.GID_filename), 'r')
        self.GIDs = []
        for l in GID_filename:
            a = l.split()
            self.GIDs.append([int(a[0]), int(a[1])])
        GID_filename.close()


        # population sizes
        self.N_X = [self.GIDs[i][1] - self.GIDs[i][0] + 1
                          for i in range(len(self.GIDs))]


        # discard population name 'TC' if no thalamic neurons are simulated
        self.X = self.X[:len(self.N_X)]


        if RANK == 0:
            print('\nGlobal uncorrected ids (first id, last id):')
            for GID in self.GIDs:
                print(GID)
            print('\n')

            print('Population sizes:')
            for X, s in zip(self.X, self.N_X):
                print('{0}:\t{1}'.format(X, s))
            print('\n')

            # total number of neurons in the simulation
            print('Total number of neurons:')
            print(sum(self.N_X))
            print('\n')


    def run(self):
        '''
        Default procedure for converting file output from NEST
        '''
        # clear destination of processed nest output
        if RANK == 0:
            if os.path.isdir(self.output_path):
                pass
            else:
                os.mkdir(self.output_path)
        #sync
        COMM.Barrier()

        # load spikes from gdf files, correct GIDs,
        # merge them in separate files, and store spike trains
        self.GIDs_corrected = self.get_GIDs()

        #combine spike files generated on individual threads into one file
        #per population
        self.merge_files(self.spike_detector_label)

        #get the positions, correct the GIDs and write to file:
        self.positions_corrected = self.get_positions()

        #get the spike data, fill in dict
        self.spike_detector_output = {}
        for i, X in enumerate(self.X):
            self.spike_detector_output[X] = read_gdf(os.path.join(
                                        self.output_path,
                                        self.spike_detector_label + X + '.gdf'))


    def get_GIDs(self):
        '''
        NEST produces one population spike file per virtual process.
        This function returns the corrected GIDs,
        i.e. first corrected GID and population size of each population
        '''
        fname = os.path.join(self.output_path, 'population_GIDs.json')
        if RANK == 0:
            #load json file if it exist
            if os.path.isfile(fname):
                with open(fname, 'rb') as fp:
                    GIDs = json.load(fp)
            else:
                raw_first_gids = [self.GIDs[i][0]
                                  for i in range(len(self.X))]
                converted_first_gids = [int(1 + np.sum(self.N_X[:i]))
                                            for i in range(len(self.X))]

                GIDs = {}
                for i, X in enumerate(self.X):
                    GIDs[X] = [converted_first_gids[i], self.N_X[i]]
                # write dict with corrected gids to file

                print('Writing corrected population GIDs to file:')
                print('writing:\t{0}'.format(fname))
                with open(fname, 'wb') as fp:
                    json.dump(GIDs, fp)
                print('\n')

                print('Global corrected ids (X, first id, N_X):')
                for X, GID in list(GIDs.items()):
                    print('{0}:\t{1}'.format(X, GID))
                print('\n')
        else:
            GIDs = None

        return COMM.bcast(GIDs, root=0)


    def merge_files(self, detector_label):
        '''
        NEST produces one population spike/voltage file per virtual process.
        This function gathers and combines them into one single file per
        population.

        Arguments
        ---------
        detector_label : str
        '''

        if RANK == 0:
            print('Writing spikes/voltages with corrected GIDs to file:')

        for pop_idx, pop in enumerate(self.X):
            #parallelize on the population level
            if pop_idx % SIZE == RANK:
                files = glob.glob(os.path.join(self.input_path,
                                               detector_label \
                                               + str(pop_idx) + '.gdf'))

                gdf = []
                for f in files:
                    new_gdf = read_gdf(f)
                    # spike files: GID, spike time
                    # voltage files: GID, voltage probe time, voltage
                    for el in new_gdf:
                        # correct GID
                        el[0] = self.correct_any_GID(el[0])
                        # discard spike times < transient,
                        # spike times now start at 0.
                        el[1] -= self.TRANSIENT
                        if el[1] >= 0:
                            gdf.append(el)

                if 'voltages' in detector_label:
                    # sort for neuron ids (along the first axis)
                    gdf = sorted(gdf, key=lambda x: x[0])

                print('writing: %s' % os.path.join(self.output_path,
                    detector_label + '%s.gdf' % pop))
                write_gdf(gdf, os.path.join(self.output_path,
                    detector_label + '%s.gdf' % pop))
        COMM.Barrier()

 
    def correct_any_GID(self, aGID):
        '''
        Needs self.GIDs and self.GIDs_corrected. Checks to which population a
        GID belongs and corrects it

        Arguments
        ---------
        aGID : int,
            index of a GID in population
        '''
        for i, X in enumerate(self.X):
            if (aGID >= self.GIDs[i][0]) and (aGID <= self.GIDs[i][1]):
                GID_corrected = self.GIDs_corrected[X][0] + aGID \
                                - self.GIDs[i][0]
                break
            else:
                continue
        return GID_corrected


    def get_positions(self):
        '''Get the neuron positions from file.'''
        # convert lists to a nicer format, i.e., [[L23E, L23I], []....]
        print('\nLoading positions from file')
        #do processing on RANK 0, but
        #consider parallel implementation, however procedure is fairly fast
        if RANK == 0:
            fname = os.path.join(self.output_path, 'all_positions.h5')
            #position files
            files = glob.glob(os.path.join(self.input_path,
                                           self.position_filename_label \
                                           + '*.dat'))
            #iterate over files
            for i, f in enumerate(files):
                if i == 0:
                    positions = read_gdf(f)
                else:
                    positions = np.r_[positions, read_gdf(f)]

            #sort according to GID
            positions = positions[positions[:, 0].argsort()]

            #correct GIDs
            for i, pos in enumerate(positions):
                positions[i][0] = self.correct_any_GID(positions[i][0])

            #dict data container
            all_positions_dict = {}

            #hdf5 container
            all_positions = h5py.File(fname)

            #fill in values
            for i, X in enumerate(self.X):
                X_pos = []
                for j, pos in enumerate(positions):
                        # if GID belongs to population X
                        if (pos[0] >= self.GIDs_corrected[X][0]) and \
                            (pos[0] <= sum(self.GIDs_corrected[X])-1):
                            X_pos.append(pos.tolist())
                all_positions_dict[X] = np.array(X_pos)

                #dump to h5 if entry doesn't exist:
                if X not in list(all_positions.keys()):
                    # each dataset has a fixed type -> here: GIDs have become
                    # floats
                    dset = all_positions.create_dataset(X,
                                                        data=X_pos,
                                                        compression='gzip',
                                                        compression_opts=2)

            all_positions.close()
            print('Positions loaded.')

        else:
            all_positions_dict = None
        return COMM.bcast(all_positions_dict, root=0)


    def compute_position_hist(self, positions):
        '''
        assign the neurons to spatial bins

        Arguments
        ---------
        positions : np.ndarray
            columns are [neuron #, x-pos, y-pos]

        Returns
        -------
        pos_hist : np.ndarray
            histogram over neuron positions
        '''
        pos_hist = np.histogram2d(positions[:, 1].astype(float),
                                          positions[:, 2].astype(float),
                                  bins=[self.pos_bins, self.pos_bins])[0]
        return pos_hist.astype(int)


    def compute_time_binned_sptrains(self, X, gdf, time_bins, dtype=np.uint16):
        '''
        Compute the spike train histograms

        Arguments
        ---------
        X : str
            population name
        gdf : np.ndarray
            colums are [GID, spike_time]
        time_bins : np.ndarray
            bin array for histogram
        dtype : type(int)
            any integer type that will fit data.

        Returns
        -------
        out : scipy.sparse.csr.csr_matrix
            sparse pop-size times time_bins.size spike rate histogram
        '''
        #need one extra bin
        dt = np.diff(time_bins)[0]
        time_bins_h = np.r_[time_bins, [time_bins[-1] + dt]] - self.dt/2.

        #spike train histogram, use scipy.sparse.lil_matrix
        #(row-based linked list sparse matrix).
        #lil_matrix supports slicing which coo_matrix do not, so we convert to
        #coo (COOrdinate format) later on before saving to file
        sptrains = sp.lil_matrix((self.N_X[self.X.index(X)],
                                  time_bins_h.size), dtype=dtype)

        #if no spikes, return empty array
        if gdf.size == 0:
            return sptrains.tocsr()
        else:
            # get indices of the time bins to which each spike time belongs
            inds = (gdf[:, 1] >= time_bins_h[0]) & \
                   (gdf[:, 1] <= time_bins_h[-1])
            pos_t = np.digitize(gdf[inds, 1].astype(np.float), time_bins_h[1:])
            #raise Exception
            #create COO matrix
            sptrains = sp.coo_matrix((np.ones(inds.sum(), dtype=dtype),
                                      (gdf[inds, 0]-self.GIDs_corrected[X][0],
                                       pos_t)),
                shape=sptrains.shape, dtype=dtype)

            #we are not changing the matrix later, so convert to CSR_matrix
            #that supports faster iteration
            return sptrains.tocsr()


    def compute_pos_binned_sptrains(self, positions, sptrains, dtype=np.uint8):
        '''
        compute the position-binned spike trains

        Arguments
        ---------
        positions : np.ndarray
            neuron positions
        sptrains : np.ndarray
            neuron spike trains
        dtype : type(int)
            integer type that can fit data

        Returns
        -------
        pos_binned_sptrains : np.ndarray
            pos x pos x spike rate 3D spike rate array

        '''
        #match position indices with spatial indices
        pos_x = np.digitize(positions[:, 1].astype(float), self.pos_bins[1:],
                            right=True)
        pos_y = np.digitize(positions[:, 2].astype(float), self.pos_bins[1:],
                            right=True)

        #2D sparse array with spatial bins flattened to 1D
        map_y, map_x = np.mgrid[0:self.pos_bins.size-1, 0:self.pos_bins.size-1]
        map_y = map_y.ravel()
        map_x = map_x.ravel()

        sptrains_coo = sptrains.tocoo().astype(dtype)
        nspikes = np.asarray(sptrains_coo.sum(axis=1)).flatten()
        data = sptrains_coo.data
        col = sptrains_coo.col
        row = np.zeros(sptrains_coo.nnz)
        j = 0
        for i, n in enumerate(nspikes):
            [ind] = np.where((map_x == pos_x[i]) & (map_y==pos_y[i]))[0]
            row[j:j+n] = ind
            j += n

        binned_sptrains_coo = sp.coo_matrix((data, (row, col)),
                                            shape=(map_x.size,
                                                   sptrains.shape[1]))

        try:
            assert(sptrains.sum() == binned_sptrains_coo.tocsr().sum())
        except AssertionError as ae:
            raise ae, \
                'sptrains.sum()={0} != binned_sptrains_coo.sum()={1}'.format( \
                    sptrains.sum(), binned_sptrains_coo.tocsr().sum())

        return binned_sptrains_coo.tocsr()


    def print_firing_rates(self):
        '''
        compute and print the firing rates from spike detector output

        Returns
        -------
        output : list,
            each list element a length 2 list of population name and rate

        '''
        if RANK == 0:
            rates = []
            for i, X in enumerate(self.X):
                if self.spike_detector_output[X].size > 0:
                    rates +=  [float(len(self.spike_detector_output[X])) \
                             / self.N_X[i] / (self.t_sim-self.TRANSIENT) * 1E3]
                else:
                    rates += [0]

            outp = list(zip(self.X, rates))

            print('\nFiring rates:')
            for X, r in outp:
                print('{0}:\t{1}'.format(X, r))
            print('\n')
        else:
            outp = None
        return COMM.bcast(outp, root=0)


    def compute_binned_rates(self, binned_spcounts, pos_hist):
        '''
        Compute the spatially binned spike rates

        Arguments
        ---------
        binned_spcounts : np.ndarray
            spatiotemporally binned counts of spikes
        pos_hist : np.ndarray
            histogram over bin positions

        Returns
        -------
        binned_sprates : np.ndarray
            averaged spike rates over spatial bins

        '''
        #spatial spike rates, avoid division by zero, start TRANSIENT is removed
        binned_sprates = np.zeros(binned_spcounts.shape)
        binned_sprates[pos_hist != 0] = binned_spcounts[pos_hist != 0].astype(
            float) / pos_hist[pos_hist != 0]
        binned_sprates *= 1E3
        binned_sprates /= (self.t_sim - self.TRANSIENT)

        return binned_sprates


    def compute_binned_sprates(self, binned_sptrains, pos_hist, time_bins):
        '''
        compute the the time and position downsampled spike rates averaged over
        neurons in a spatial bin

        Arguments
        ---------
        binned_sptrains : np.ndarray
            time-binned spike data
        pos_hist : np.ndarray
            histogram over neuron positions
        time_bins : np.ndarray
            array over timebins

        Returns
        -------
        binned_sprates : np.ndarray
            time and position-binned spike rate

        '''
        ##avoid division by zero, flatten for array division
        pos_hist = pos_hist.flatten()
        pos_inds = np.where(pos_hist > 0)
        #pos_denom = pos_hist[pos_mask]

        pos_hist_diag = sp.diags(pos_hist.flatten(), 0)

        #2D-array for instantaneous rate, shaped as input
        binned_sprates = binned_sptrains.astype(float)

        #compute rate profile as:
        #(spike train per patch) / (neurons per patch) * 1000 ms/s / dt
        #hence we do not divide by duration but account for dt
        binned_sprates *= 1E3
        binned_sprates /= self.BINSIZE_TIME

        binned_sprates = binned_sprates.tolil()
        binned_sprates[pos_inds] = (binned_sprates[pos_inds].toarray().T \
                                    / pos_hist[pos_inds]).T

        return binned_sprates.tocsr()


if __name__ == '__main__':
    # input and output path can be provided on the command line:
    #    python nest_preprocessing.py out_raw out_proc
    # if no argument is given, default values are used
    if len(sys.argv) != 3:
        input_path = 'out_raw'
        output_path = 'out_proc'
    else:
        input_path = sys.argv[-2]
        output_path = sys.argv[-1]

    #tic toc
    tic = time()

    preprocess = ViolaPreprocessing( input_path=input_path,
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
                                BINSIZE_AREA=0.1,
    )


    #run through the main nest output files, collect GDFs, spikes and positions
    preprocess.run()

    # firing rates
    preprocess.print_firing_rates()


    #iterate over all network populations
    print('Computing results of pop: ')
    #simple attempt at loadbalancing, also across compute nodes:
    argsort = np.argsort(preprocess.N_X)[::-1]
    for i, j in enumerate(argsort):
        X = preprocess.X[j]
        if i % SIZE == RANK:
            print('population {0} on RANK {1}'.format(X, RANK))

            ####################################################################
            # COMPUTE STUFF
            ####################################################################
            #get the spike output of neurons with corrected GIDs
            spikes = read_gdf(os.path.join(preprocess.output_path,
                                           preprocess.spike_detector_label +
                                           X + '.gdf'))

            #positions
            binned_positions = preprocess.compute_position_hist(
                preprocess.positions_corrected[X])

            ##position binned spike trains
            sptrains = preprocess.compute_time_binned_sptrains(
                X, spikes, preprocess.time_bins, dtype=np.uint8)
            binned_sptrains = preprocess.compute_pos_binned_sptrains(
                preprocess.positions_corrected[X], sptrains, dtype=np.uint16)

            #spike counts and rates per spatial bin
            binned_spcounts = np.asarray(
                binned_sptrains.sum(axis=1).astype(int).reshape((
                    preprocess.pos_bins.size-1, -1)))
            binned_sprates = preprocess.compute_binned_rates(binned_spcounts,
                                                             binned_positions)

            #create some time resampled spike and rate histograms
            sptrains_rs = preprocess.compute_time_binned_sptrains(
                X, spikes, preprocess.time_bins_rs, dtype=np.uint8)
            binned_sptrains_rs = preprocess.compute_pos_binned_sptrains(
                preprocess.positions_corrected[X], sptrains_rs, dtype=np.uint16)
            #spatially binned resampled spike trains
            binned_sprates_rs = preprocess.compute_binned_sprates(
                binned_sptrains_rs, binned_positions, preprocess.time_bins_rs)


            ####################################################################
            # WRITE STUFF
            ####################################################################

            filenames = [
                'all_binned_sprates_rs_{0}.h5'
            ]

            datasets = [
                binned_sprates_rs
            ]

            for fname, data in zip(filenames, datasets):
                while True:
                    try:
                        fpath = os.path.join(preprocess.output_path,
                                             fname.format(X))
                        f = h5py.File(fpath, 'w')
                        dump_sparse_to_h5(X, f=f, data=data, compression='gzip',
                                          compression_opts=2)
                        f.flush()
                        f.close()
                        break

                    except: # IOError as e:
                        print("Expected bad hdf5 behaviour:", sys.exc_info(),
                              fpath)
                        sleep(1.)

            # write compact format for viola
            #Format: [x-coordinate\ y-coordinate\ timestep\ value]
            filenames = [
                'binned_sprates_rs_{0}.dat',
            ]

            datasets = [
                binned_sprates_rs,
            ]

            #spatial bin remapping to 2D:
            y, x = np.mgrid[0:np.sqrt(data.shape[0]):1,
                            0:np.sqrt(data.shape[0]):1]
            x = x.flatten()
            y = y.flatten()
            for fname, data in zip(filenames, datasets):
                fname = os.path.join(preprocess.output_path, fname.format(X))
                data = data.tocoo()
                outp = np.c_[x[data.row], y[data.row], data.col, data.data]
                np.savetxt(fname, outp, fmt=('%i', '%i', '%i', '%f'))


    COMM.Barrier()

    #put certain output data as generated on different parallel threads in }
    #single files for further analysis.
    if RANK == 0:
        filenames = [
            'all_binned_sprates_rs'
        ]

        for fname in filenames:
            fpath1 = os.path.join(preprocess.output_path, fname + '.h5')
            try:
                f1 = h5py.File(fpath1, 'w')
                for X in preprocess.X:
                    fpath0 = os.path.join(preprocess.output_path, fname + \
                                          '_{0}.h5'.format(X))
                    f0 = h5py.File(fpath0, 'r')
                    if X in f0.keys(): # quick check if key exists
                        f1.copy(f0[X], X)
                    f0.close()
                    os.system('rm {0}'.format(fpath0))
                f1.close()
            except IOError as ioe:
                f1.close()
                os.remove(fpath1)


        # write configuration file for this dataset for VIOLA
        sp = glob.glob(os.path.join(preprocess.output_path,
                                    'binned_sprates_rs_*.dat'))
        spikesFiles = ','.join([os.path.basename(sp[i]) \
                                for i in range(len(sp))])

        popColors = []
        cmap = plt.get_cmap('rainbow_r', len(preprocess.X))
        for i in range(cmap.N):
            rgb = cmap(i)[:3]
            col_hex = mpc.rgb2hex(rgb)
            popColors.append(col_hex)
        popColors = ','.join(popColors)

        config_dict = {}
        config_dict.update({
            "popNum": len(preprocess.X),
            "popNames": ','.join(preprocess.X),
            "spikesFiles": spikesFiles,
            "timestamps": int((preprocess.t_sim - preprocess.TRANSIENT) \
                              / preprocess.BINSIZE_TIME),
            "resolution": preprocess.BINSIZE_TIME,
            "xSize": preprocess.extent_length,
            "ySize": preprocess.extent_length,
            "dataType": "binned",
            "xBins": int(preprocess.extent_length / preprocess.BINSIZE_AREA),
            "yBins": int(preprocess.extent_length / preprocess.BINSIZE_AREA),
            "xLFP": 10,
            "yLFP": 10,
            "timelineLenght": 40,
            "popColors": popColors,
        })

        with open(os.path.join(preprocess.output_path, \
                               'config_proc.json'), 'w') as f:
            json.dump(config_dict, f)


    COMM.Barrier()
    if RANK == 0:
        toc = time()-tic
        print('analysed in {0} seconds'.format(toc))
