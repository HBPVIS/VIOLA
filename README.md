# VIOLA

VIOLA (VIsualization Of Layer Activity) is an interactive, web-based tool
to visualize activity data in multiple 2D layers such as
the simulation output of neuronal networks with 2D geometry.

A usage example demonstrates the visualization of spike data resulting from a
[NEST](http://nest-simulator.org) simulation of a spatially structured
point-neuron network with excitatory and inhibitory neuron populations and an
external stimulus.

## Getting started

### 1. Get test data

Spatially resolved time series data to be visualized with VIOLA can have two
different formats:
* **raw:** spike times associated with spatial locations of spiking neurons
* **preprocessed:** spatially and temporally binned spike data including LFP
signals

We have prepared one data set of each format that can be downloaded here:
* [RawData.zip](https://hbpvis.github.io/VIOLA/downloads/RawData.zip)
* [PreprocessedData.zip](https://hbpvis.github.io/VIOLA/downloads/PreprocessedData.zip)
(recommended to start with for testing out VIOLA)

Extract the archived files, e.g., using `unzip PreprocessedData.zip`.  
Each data set contains text files with the data to be visualized and a
corresponding configuration file for VIOLA:

#### Configuration files

raw                    | preprocessed
---------------------- | -----------------
config_raw.json        | config_proc.json

#### Data files

raw                    | preprocessed
---------------------- | -----------------
spikes-0.gdf           | binned_sprates_rs_EX.dat
spikes-1.gdf           | binned_sprates_rs_IN.dat
spikes-2.gdf           | binned_sprates_rs_STIM.dat
neuron_positions-0.dat | LFPdata.lfp
neuron_positions-1.dat |
neuron_positions-2.dat |

Alternatively, you can generate test data yourself as described in
[Generating test data](#generating-test-data-optional).

### 2. Start VIOLA

VIOLA runs in a web browser and the preferred browser is
[Chrome](https://www.google.de/chrome).

Start VIOLA from its [GitHub Page](http://hbpvis.github.io/VIOLA).  
Note that this version of the tool may differ from the current master branch of
this repository.  
To get the latest version, you can clone the repository (e.g.,
`git clone https://github.com/HBPVIS/VIOLA.git`),
navigate to the directory **VIOLA**, and open the contained file index.html in
the browser.

Upon startup, VIOLA opens the **Setup Page** to configure the visualization for
a specific data set.  
Just upload the configuration file (config_proc.json for preprocessed data), and
then click the button `Setup visualization` to get to the **Main Page**.

Using the **Upload Panel** to the left, you can upload all files for the data
(see the [Tables](#configuration-files) above) to be visualized:  
either by dragging and dropping the files to the blue box or by opening a file
manager with the corresponding button.  
If the background color of the field for a data file changes from red to green,
the upload has been successful.  
As soon as all files are uploaded, close the upload panel by clicking the **x**
at its top right corner.

Press **Play** to start the visualization.

For further documentation, please refer to the
[VIOLA Wiki](https://github.com/HBPVIS/VIOLA/wiki) containing the
[User Manual](https://github.com/HBPVIS/VIOLA/wiki/VIOLA-User-Manual)
and the
[Developer Manual](https://github.com/HBPVIS/VIOLA/wiki/VIOLA-Developer-Manual).

## Generating test data (optional)

The scripts to simulate a spatially structured network of spiking point-neurons
are in the directory **test_data** in this repository.  
Simulations rely on the simulator [NEST](http://nest-simulator.org) and are
implemented using the [Python](http://www.python.org) interface.  
Software dependencies for the simulation scripts are summarized together with
the tested version numbers [below](#software-dependencies).

For generating **raw data**, run

    python topo_brunel_alpha_nest.py out_raw

The created directory **out_raw** contains configuration and data files for raw
data and the [Tables](#configuration-files) above indicate which files need to
be uploaded to VIOLA for visualization.

Having generated raw data, you can generate **preprocessed data** by running

    python nest_preprocessing.py out_raw out_proc

To speed up the preprocessing step, you can also use OpenMPI und run instead

    mpirun -np 2 python nest_preprocessing.py out_raw out_proc

Afterwards, generate LFP data. You need to compile the
[NEURON](https://neuron.yale.edu) model once on your system by
executing

    nrnivmodl alphaisyn.mod

and then you can run

    python fake_LFP_signal.py out_raw out_proc

The created directory **out_proc** contains configuration and data files for
preprocessed data and the [Tables](#configuration-files) above indicate which
files need to be uploaded to VIOLA for visualization.

### Software dependencies
* [NEST](http://nest-simulator.org) v2.10.0
* [NEURON](https://neuron.yale.edu) v7.4
* [Python](http://www.python.org) v2.7.11
  * numpy v1.10.14
  * matplotlib v2.0.2
  * scipy v0.17.0
  * h5py v2.5.0
  * mpi4py v2.0.0
  * LFPy v2.0.0
  * quantities v0.12

## Authors

* Corto Carde, Johanna Senk, Espen Hagen, Benjamin Weyers
