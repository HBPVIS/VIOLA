# VIOLA

## Tool description

---
**VIOLA (VIsualizer Of Layer Activity)**
is a tool to visualize activity in multiple 2D layers in an interactive and
efficient way. It gives an insight into spatially resolved time series
such as, for example, simulation results of neural networks with geometry.

The usage example shows how VIOLA can be used to visualize spike data from a
NEST simulation ( http://nest-simulator.org/ ) of an excitatory and an
inhibitory neuron population with distance-dependent connectivity..

Detailed documentation can be found in the 'User Manual' and in the 'Developer
quick guide'.

## Usage example

---
### 1. Generate spike data with NEST

    python topo_brunel_alpha_nest.py

creates a directory 'out_raw' which contains the simulation output and a
configuration file for VIOLA

### 2. Visualize raw spike data

- start VIOLA and load the configuration file 'config_raw.json'
- upload the following files:
  - spikes-0.gdf, spikes-1.gdf
  - neuron_positions-0.dat, neuron_positions-1.dat

### 3. Preprocess spike data

    python viola_nest_preprocessing.py out_raw out_proc

applies a spatial binning and changes the time step, output is stored in
'output_proc'

### 4. Generate LFP like signal from spike data

    python viola_fake_LFP_signal.py out_raw out_proc

### 5. Visualize preprocessed data
- start VIOLA and load the configuration file 'config_proc.json'
- upload the following files:
  - binned_sprates_rs_EX.dat, binned_sprates_rs_IN.dat
  - LFPdata.lfp

## Dependencies (tested with)

---
- NEST 2.8

## Authors

---
- Corto Carde, Johanna Senk, Espen Hagen, Benjamin Weyers

