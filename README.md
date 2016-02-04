# VIOLA

## Tool description

**VIOLA (VIsualizer Of Layer Activity)**
is a tool to visualize activity in multiple 2D layers in an interactive and
efficient way. It gives an insight into spatially resolved time series
such as, for example, simulation results of neural networks with 2D geometry.

The usage example shows how VIOLA can be used to visualize spike data from a
NEST simulation (http://nest-simulator.org/) of an excitatory and an
inhibitory neuron population with distance-dependent connectivity.

Detailed documentation will be made available in the Wiki. Short usage example
are shown below.

## Usage example
The following description includes generation and processing of output data from
a simulation of a point-neuron network with 2D geometry implemented using NEST
through Python (http://www.python.org). The scripts can be found in the folder
'test_data'. These steps can be skipped, as the simulation output can already
be found in 'test_data/out_raw' and 'test_data/out_proc'.

### 1. Generate spike data with NEST

    python topo_brunel_alpha_nest.py out_raw

creates a directory 'out_raw' which contains the simulation output and a
configuration file for VIOLA.

### 2. Visualize raw spike data

- start VIOLA (/VIOLA/index.html) in a browser (Chrome preferred), and load the
configuration file 'config_raw.json'. Alternatively, adjust the parameters
manually.
- upload the following files:
  - spikes-0.gdf, spikes-1.gdf
  - neuron_positions-0.dat, neuron_positions-1.dat

### 3. Preprocess spike data

    python viola_nest_preprocessing.py out_raw out_proc

or with OpenMPI:

    mpirun -np 2 python viola_nest_preprocessing.py out_raw out_proc

applies a spatial binning and changes the time step, output files are stored in
'out_proc'.

### 4. Generate LFP like signal from spike data

    python viola_fake_LFP_signal.py out_raw out_proc

### 5. Visualize preprocessed data
- start VIOLA and load the configuration file 'config_proc.json'
- upload the following files:
  - binned_sprates_rs_EX.dat, binned_sprates_rs_IN.dat
  - LFPdata.lfp

## Simulation script dependencies

- Python 2.7.x
- numpy
- matplotlib
- scipy
- h5py
- mpi4py
- NEST 2.10.x

## VIOLA-compatible browsers

- Chrome
- Opera

## Authors

- Corto Carde, Johanna Senk, Espen Hagen, Benjamin Weyers
