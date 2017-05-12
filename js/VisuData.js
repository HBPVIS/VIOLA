if (Visu === undefined)
  var Visu = {};

Visu.Data = function(params) {

  //Data type independant parameters
  this.xSize = parseInt(params.x);
  this.ySize = parseInt(params.y);

  this.timestamps = parseInt(params.timestamps);
  this.resolution = parseFloat(params.resolution); //In ms

  this.nLayers = params.layers; //Number of layers
  this.layerNames = params.names;
  this.layerColors = params.colors;
  this.spikesFiles = params.spikes;

  this.mReady = []; //Boolean to know if spikes data is ready

  //Defines the structure of the data
  this.dataType = params.dataType;

  if (this.dataType == "binned") {
    //Size of a population
    this.xNeurons = parseInt(params.xn);
    this.yNeurons = parseInt(params.yn);
    for (var i = 0; i < this.nLayers; i++) {
      this.mReady[i] = false;
    };
  } else if (this.dataType == "neuron") {
    //Booleans to check if positions of neurons have been loaded
    this.posReady = [];
    //Array with neurons and their positions
    this.neuronPos = [];
    //Array with indices of neurons belonging in the layers;
    this.neuronLayers = [];
    this.neuronSpikes = [];
    this.posFiles = params.positions;
    for (var i = 0; i < this.nLayers; i++) {
      this.neuronSpikes[i] = [];
      this.neuronPos[i] = [];
      this.neuronLayers[i] = [];
      this.mReady[i] = false;
      this.posReady[i] = false;
    };
  };

  //this.timestamps = 6471;//6470 limit

  this.simulationLength = this.timestamps * this.resolution;
  this.currTime = 0;

  this.maxSpikesAmount = 0;
  this.maxSumAmount = 0;
  this.maxTotalAmount = 0;

  //LFP
  this.lfpXSize = parseInt(params.lfpx);
  this.lfpYSize = parseInt(params.lfpy);
  this.lfpReady = false;
  this.lfpDataset = [];
  this.lfpMax = 0;
  this.lfpMin = 0;
  this.lfpWidth = 0;

  //3D Timeline
  this.zTimeSize = parseInt(params.ztimesize);

  this.datasets = [];

  //Data Scaling
  this.dataOffset = 0;
  this.dataMultiply = 1;
  this.dataPower = 1;
  this.rms = 1;

  this.sumDatasets = [];

  this.totalDatasets = [];

  for (var i = 0; i < this.nLayers; i++) {
    this.datasets[i] = [];
    this.sumDatasets[i] = [];
    this.totalDatasets[i] = [];
    for (var j = 0; j < this.timestamps; j++) {
      this.datasets[i][j] = new Uint32Array(this.xNeurons * this.yNeurons);
      this.sumDatasets[i][j] = new Uint32Array(this.xNeurons + this.yNeurons);
      this.totalDatasets[i][j] = 0;
      for (var k = 0; k < this.xNeurons + this.yNeurons; k++) {
        this.sumDatasets[i][j][k] = 0;
      };
    };
  };

  for (var i = 0; i < this.timestamps; i++) {
    this.lfpDataset[i] = [];
  };
};

Visu.Data.prototype = {

  setTime: function(v) {
    this.currTime = Math.floor(v * 10) / 10;
  },

  computeRMS: function() {
    var sum = 0,
      num = 0;
    for (var i = 0; i < this.nLayers; i++) {
      if (this.mReady[i]) {
        num++;
        for (var j = 0; j < this.timestamps; j++) {
          for (var k = 0; k < this.xNeurons * this.yNeurons; k++) {
            sum += this.datasets[i][j][k] * this.datasets[i][j][k];
          };
        };
      };
    };

    sum /= num * this.timestamps * this.xNeurons * this.yNeurons;

    this.rms = Math.sqrt(sum);

  },

  computeCrossCorrelation: function(l1, x1, y1, l2, x2, y2, max_delay) {
    var s = [],
      m1 = 0,
      m2 = 0,
      d1 = 0,
      d2 = 0,
      denom;


    //Init result vector
    for (var i = 0; i < 2 * max_delay; i++) {
      s[i] = 0;
    };

    //Compute the mean of the series
    for (var i = 0; i < this.timestamps; i++) {
      m1 += this.datasets[l1][i][x1 * this.yNeurons + y1];
      m2 += this.datasets[l2][i][x2 * this.yNeurons + y2];
    };

    m1 /= this.timestamps;
    m2 /= this.timestamps;

    //Compute the normalizing factor
    for (var i = 0; i < this.timestamps; i++) {
      d1 += (this.datasets[l1][i][x1 * this.yNeurons + y1] - m1) *
              (this.datasets[l1][i][x1 * this.yNeurons + y1] - m1);
      d2 += (this.datasets[l2][i][x2 * this.yNeurons + y2] - m2) *
              (this.datasets[l2][i][x2 * this.yNeurons + y2] - m2);
    };

    denom = Math.sqrt(d1 * d2);

    if (denom == 0) {
      return s;
    };

    //Compute the cross-correlation factor serie
    for (var delay = -max_delay, index = 0; delay < max_delay; delay++, index++) {
      s[index] = 0;
      for (var i = 0, j = 0; i < this.timestamps; i++) {
        j = i + delay;
        while (j < 0) {
          j += this.timestamps;
        }
        j %= this.timestamps;
        s[index] += (this.datasets[l1][i][x1 * this.yNeurons + y1] - m1) *
                      (this.datasets[l2][j][x2 * this.yNeurons + y2] - m2);
      }
      s[index] /= denom;
      if (s[index] < 0) {
        s[index] = 0;
      };
    };

    return s;
  },

  importPos: function(d, s) {
    d = d.split("\n");
    var temp;
    for (var i = 0; i < d.length; i++) {
      temp = d[i].split(" ");
      if (temp[0] != "") {
        this.neuronPos[temp[0]] = [temp[1], temp[2]];
        this.neuronLayers[s][i] = temp[0];
      };
    };
  },

  importSpikes: function(d, s) {
    d = d.split("\n");
    var time, neuron, cache;
    for (var i = 0; i < d.length; i++) {
      cache = d[i].split("\t");
      neuron = parseInt(cache[0]);
      time = Math.round(parseFloat(cache[1]) / this.resolution);
      if (!isNaN(time)) {
        if (this.neuronSpikes[s][time] == undefined) {
          this.neuronSpikes[s][time] = [];
        };
        this.neuronSpikes[s][time].push(neuron);
      };
    };
  },

  dataCompact2Gen: function(d, s) {
    //Split to have 1 neuron per cell
    d = d.split("\n");
    var data, x, y, time, l;

    //Put corresponding int in datasets table
    for (var i = 0; i < d.length; i++) {
      l = d[i].split(" ");
      x = parseInt(l[0]);
      y = parseInt(l[1]);
      time = parseInt(l[2]);
      data = parseInt(l[3]);
      if (time < this.timestamps) {
        this.datasets[s][time][x * this.yNeurons + y] = data;
        this.sumDatasets[s][time][x] += data;
        this.sumDatasets[s][time][this.xNeurons + y] += data;
        this.totalDatasets[s][time] += data;
        if (data > this.maxSpikesAmount)
          this.maxSpikesAmount = data;
      };
    };

   // 30% of maxSpikesAmount is used as cut-off limit
   // for color scales and voxel sizes
   this.maxSpikesCutoff = 0.3 * this.maxSpikesAmount;

    for (var i = 0; i < this.timestamps; i++) {
      if (this.totalDatasets[s][i] > this.maxTotalAmount)
        this.maxTotalAmount = this.totalDatasets[s][i];
      for (var j = 0; j < this.xNeurons + this.yNeurons; j++) {
        if (this.sumDatasets[s][i][j] > this.maxSumAmount)
          this.maxSumAmount = this.sumDatasets[s][i][j];
      };
    };
  },

  lfpDataGen: function(d) {
    d = d.split("\n");
    var data, i, j, k;
    var mean = [];
    this.stdDev = 0;
    for (i = 0; i < this.lfpYSize * this.lfpXSize; i++) {
      mean[i] = 0;
      d[i] = d[i].split(",");
      for (k = 0; k < this.timestamps; k++) {
        data = Number(d[i][k] || "0");
        this.lfpDataset[k][i] = data;
        mean[i] += data;
      };
      mean[i] /= this.timestamps;
    };

    //The LFP signals are offset to have a mean of 0
    //The stdDev value is used to adjust the color range to the LFP values
    for (i = 0; i < this.lfpYSize * this.lfpXSize; i++) {
      for (k = 0; k < this.timestamps; k++) {
        this.lfpDataset[k][i] -= mean[i];

        this.stdDev += this.lfpDataset[k][i] * this.lfpDataset[k][i];

        if (this.lfpDataset[k][i] > this.lfpMax)
          this.lfpMax = this.lfpDataset[k][i];
        if (this.lfpDataset[k][i] < this.lfpMin)
          this.lfpMin = this.lfpDataset[k][i];
      };
    };

    this.stdDev /= this.lfpYSize * this.lfpXSize * this.timestamps;
    this.stdDev = Math.sqrt(this.stdDev);

    this.lfpWidth = 2 * this.stdDev;
  },

  updateStdDev: function(v) {
    this.lfpWidth = this.stdDev * parseFloat(v);
    updated = true;
  },

  updateScalingFactor: function(v) {
    var val = parseFloat(v);
    if (val > 0)
      this.dataMultiply = val;
    else
      alert("Please use a number greater than 0");
    updated = true;
  },

  updateOffsetFactor: function(v) {
    this.dataOffset = parseFloat(v);
    updated = true;
  },

  updatePowerFactor: function(v) {
    this.dataPower = parseFloat(v);
    updated = true;
  },

  getScaledValue: function(v) {
    if (v == 0)
      return 0;
    else
      return this.dataMultiply * v - this.dataOffset;
  },

  getUnscaledValue: function(v) {
    if (this.dataMultiply == 0)
      return 0;
    else
      return ((v + this.dataOffset) / this.dataMultiply);
  }
};
