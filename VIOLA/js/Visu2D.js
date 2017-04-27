if (Visu === undefined)
  var Visu = {};

Visu.Renderer2D = function(panel, data, name) {

  this.miniCW = 170;
  this.miniCH = 170;

  this.miniW = 120;
  this.miniH = 120;

  this.miniLW = 300;
  this.miniLH = 40;

  this.data = data;


  this.miniRectW = this.miniW / this.data.xNeurons;
  this.miniRectH = this.miniH / this.data.yNeurons;

  this.neuronDisplaySize = 4;

  var ih = '<table id="miniCanvasTable">';

  for (var i = 0; i < data.nLayers; i++) {
    if (i % 2 == 0) {
      ih += '<tr>';
    };

    ih += '<td><canvas id="mini' + i + '" width="' + this.miniCW +
      '" height="' + this.miniCH + '"></canvas></td>';

    if (i % 2 == 1) {
      ih += '</tr>';
    };
  };

  ih += '<tr><td colspan=2><div' +
        'style="margin:auto;margin-top:10px;width:310px">';

  if (this.data.dataType == "binned") {
    ih += '<canvas id="miniL" width="' + this.miniLW + '" height="' +
      this.miniLH + '"></canvas>';
  } else if (this.data.dataType == "neuron") {
    ih +=
      '<span>Scale number of neurons displayed</span><br>none ' +
      '<input onmousedown="' + name + '.disableDrag(true)" onmouseup="' + name +
      '.disableDrag(false)" type="range" min="0" max="100" value="100"' +
      'step="1" oninput="' + name +
      '.setDensity(this.value)"' +
      'style="width:200px;vertical-align:middle;display:inline;' +
      'margin: 0 10px 0">all';
  };

  ih += '</div></td></tr></table>';

  this.panel = panel;

  // wrap table in a div and place it on top of other elements
  var ihElem = document.createElement("div");
  ihElem.innerHTML = ih;
  panel.insertBefore(ihElem, panel.childNodes[0]);

  panel.ondragstart = function(e) {
    e.dataTransfer.setData("elemT", "dragPanel");
    e.dataTransfer.setData("id", e.target.id);
    e.dataTransfer.setData("parent", e.target.parentNode);

    console.log("drag of " + e.target.id);
  };

  panel.ondragover = function(e) {
    e.preventDefault();
    return false;
  };

  panel.ondrop = function(e) {
    e.preventDefault();

    if (e.dataTransfer.getData("elemT") == "dragPanel") {
      var elem = document.getElementById(e.dataTransfer.getData("id"));
      var prevParent = elem.parentNode;
      this.parentNode.appendChild(elem);
      prevParent.appendChild(this);
    };

    return false;
  };

  //Contexts for different populations of neurons' respective mini canvas
  this.mCtx = [];

  for (var c = 0; c < data.nLayers; c++) {
    this.mCtx[c] = document.getElementById("mini" + c)
      .getContext("2d");
  };

  if (data.dataType == "binned")
    this.mCtxL = document.getElementById("miniL")
    .getContext("2d");

  this.offsetW = (this.miniCW - this.miniW) / 2;
  this.offsetH = (this.miniCH - this.miniH) / 2;

  //Density display for neurons
  this.density = 100;
};


Visu.Renderer2D.prototype = {

  colorRect: function(d) {
    d = d / this.data.maxSpikesCutoff;
    if (d < 0.33)
      return "rgb(" + Math.round(3 * d * 255) + ",0,0)";
    else if (d < 0.66)
      return "rgb(255," + Math.round(255 * (d - 0.33) * 3) + ",0)";
    else if (d < 1)
      return "rgb(255,255," + Math.round(255 * (d - 0.66) * 3) + ")";
    else
      return "rgb(255,255,255)"; // white above cutoff limit
  },

  disableDrag: function(v) {
    if (v) {
      this.panel.draggable = false;
    } else {
      this.panel.draggable = true;
    }
  },

  setDensity: function(v) {
    this.density = parseInt(v);
    updated = true;
  },

  draw: function(index) {
    //Only draw mini canvas for population whose data is available
    var dataNow, displayPos, spikes;
    var offset = [this.offsetW + this.miniW / 2 - this.neuronDisplaySize / 2,
                  this.offsetH + this.miniH / 2 - this.neuronDisplaySize / 2];
    var scaling = [(this.miniW - this.neuronDisplaySize) / this.data.xSize,
                   (this.miniH - this.neuronDisplaySize) / this.data.ySize];
    for (var c = 0; c < this.data.nLayers; c++) {
      if (this.data.dataType == "binned") {
        this.mCtx[c].fillStyle = "black";
        this.mCtx[c].fillRect(this.offsetW, this.offsetH, this.miniW, this.miniH);
        if (this.data.mReady[c]) {
          for (var i = 0; i < this.data.xNeurons; i++) {
            for (var j = 0; j < this.data.yNeurons; j++) {
              dataNow = this.data.getScaledValue(this.data.datasets[c][
                index
              ][i * this.data.yNeurons + j]);
              if (dataNow > 0) {
                this.mCtx[c].fillStyle = this.colorRect(dataNow);
                this.mCtx[c].fillRect(this.offsetW + i * this.miniRectW,
                                      this.offsetH + j * this.miniRectH,
                                      this.miniRectW,
                                      this.miniRectH);
              };
            };
          };
        };
      } else if (this.data.dataType == "neuron") {
        // white background with black frame
        this.mCtx[c].fillStyle = "white";
        this.mCtx[c].fillRect(this.offsetW, this.offsetH, this.miniW, this.miniH);
        this.mCtx[c].lineWidth="1";
        this.mCtx[c].rect(this.offsetW, this.offsetH, this.miniW, this.miniH);
        this.mCtx[c].stroke();
        if (this.data.mReady[c] && this.data.posReady[c]) {
          spikes = this.data.neuronSpikes[c][index];
          if (spikes !== undefined) {
            var limit = spikes.length * this.density / 100;
            for (var i = 0; i < limit; i++) {
              displayPos = [this.data.neuronPos[spikes[i]][0],
                            this.data.neuronPos[spikes[i]][1]];
              displayPos[0] = displayPos[0] * scaling[0] + offset[0];
              displayPos[1] = displayPos[1] * scaling[1] + offset[1];
              this.mCtx[c].fillStyle = this.data.layerColors[c];
              this.mCtx[c].fillRect(displayPos[0], displayPos[1],
                                    this.neuronDisplaySize,
                                    this.neuronDisplaySize);
            };
          };
        };
      };
      // time counter
      if (c == 0){
        if (this.data.dataType == "binned") {
          this.mCtx[c].fillStyle = "white";
        } else if (this.data.dataType == "neuron") {
          this.mCtx[c].fillStyle = "black";
        }
        this.mCtx[c].fillText(this.data.currTime + "/" +
                                this.data.simulationLength + " ms",
                              this.offsetW + 2,
                              this.offsetH + 15);
       };
    };
  },

  drawMiniLegend: function() {

    // TODO color bar should actually be open to the right
    this.mCtxL.clearRect(0, 0, this.miniLW, this.miniLH);

    var limit = this.data.maxSpikesCutoff;
    var sp = [Math.round(this.data.getUnscaledValue(limit / 3)),
              Math.round(this.data.getUnscaledValue(limit * 2/3)),
              Math.round(this.data.getUnscaledValue(limit))];
    var unit = ["", "K", "M", "G", "T"];
    var uToUse = [0, 0, 0];

    for (var i = 0; i < 3; i++) {
      while (sp[i] > 1000) {
        sp[i] /= 1000;
        uToUse[i]++;
      };

      if (sp[i] < 0) {
        sp[i] = 0;
        uToUse[i] = 0;
      } else if (sp[i] < 10)
        sp[i] = Math.floor(100 * sp[i]) / 100;
      else if (sp[i] < 100)
        sp[i] = Math.floor(10 * sp[i]) / 10;
      else
        sp[i] = Math.floor(sp[i]);
    };

    var grd = this.mCtxL.createLinearGradient(10, 0, this.miniLW - 10, 0);
    grd.addColorStop(0, "rgb(0,0,0)");
    grd.addColorStop(0.33, "rgb(255,0,0)");
    grd.addColorStop(0.66, "rgb(255,255,0)");
    grd.addColorStop(1, "rgb(255,255,255)");
    this.mCtxL.fillStyle = "black";
    this.mCtxL.fillRect(10, 0, this.miniLW - 20, 10);
    this.mCtxL.fillStyle = grd;
    this.mCtxL.fillRect(10, 1, this.miniLW - 20, 8);
    this.mCtxL.fillStyle = "black";
    this.mCtxL.font = "12px sans-serif";
    this.mCtxL.textBaseline = "top";
    this.mCtxL.fillText(0, 10 - this.mCtxL.measureText(0).width / 2, 12);
    this.mCtxL.fillRect(10, 10, 1, 2);
    this.mCtxL.fillText(sp[0] + unit[uToUse[0]],
                        10 - this.mCtxL.measureText(sp[0] +
                          unit[uToUse[0]]).width / 2 + (this.miniLW - 20) / 3,
                        12);
    this.mCtxL.fillRect(10 + (this.miniLW - 20) / 3, 10, 2, 2);
    this.mCtxL.fillText(sp[1] + unit[uToUse[1]],
                        10 - this.mCtxL.measureText(sp[1] + unit[uToUse[1]])
                          .width / 2 + 2 * (this.miniLW - 20) / 3,
                        12);
    this.mCtxL.fillRect(10 + 2 * (this.miniLW - 20) / 3, 10, 2, 2);
    this.mCtxL.fillText(sp[2] + unit[uToUse[2]],
                        Math.min(this.miniLW - this.mCtxL.measureText(sp[2] +
                                   unit[uToUse[2]]).width,
                                 this.miniLW - 10 - this.mCtxL.measureText(sp[2]
                                   + unit[uToUse[2]]).width / 2),
                        12);
    this.mCtxL.fillRect(this.miniLW - 10, 0, 1, 12);
    this.mCtxL.font = "16px sans-serif";
    this.mCtxL.textBaseline = "bottom";
    this.mCtxL.fillText("Bin-wise spike count (1/s)",
                        this.miniLW / 2 -
      this.mCtxL.measureText("Bin-wise spike count (1/s)").width / 2, 40);
  },

  drawMiniLegends: function() {
    for (var c = 0; c < this.data.nLayers; c++) {
      this.mCtx[c].font = "16px sans-serif";
      this.mCtx[c].textBaseline = "top";
      this.mCtx[c].fillStyle = "black";
      this.mCtx[c].fillText(data.layerNames[c],
                            (this.miniCW - this.mCtx[c]
                               .measureText(this.data.layerNames[c]).width) / 2,
                            2);
      this.mCtx[c].font = "12px sans-serif";
      this.mCtx[c].textBaseline = "middle";
      this.mCtx[c].fillText(-this.data.ySize / 2,
                            this.offsetW - 2 - this.mCtx[c]
                              .measureText(-this.data.ySize / 2).width,
                            this.offsetH);
      this.mCtx[c].fillText(-this.data.ySize / 4,
                            this.offsetW - 2 - this.mCtx[c]
                              .measureText(-this.data.ySize / 4).width,
                            this.offsetH + this.miniH / 4);
      this.mCtx[c].fillText("0",
                            this.offsetW - 2 - this.mCtx[c].measureText("0")
                              .width,
                            this.offsetH + this.miniH / 2);
      this.mCtx[c].fillText(this.data.ySize / 4,
                            this.offsetW - 2 - this.mCtx[c]
                              .measureText(this.data.ySize / 4).width,
                            this.offsetH + 3 * this.miniH / 4);
      this.mCtx[c].fillText(this.data.ySize / 2,
                            this.offsetW - 2 - this.mCtx[c]
                              .measureText(this.data.ySize / 2).width,
                            this.offsetH + this.miniH);
      this.mCtx[c].textBaseline = "top";
      this.mCtx[c].fillText(this.data.xSize / 2,
                            this.offsetW - this.mCtx[c]
                              .measureText(this.data.xSize / 2).width / 2 +
                              this.miniW,
                            this.offsetH + this.miniH + 2);
      this.mCtx[c].fillText(this.data.xSize / 4,
                            this.offsetW - this.mCtx[c]
                              .measureText(this.data.xSize / 4).width / 2 +
                              3 * this.miniW / 4,
                            this.offsetH + this.miniH + 2);
      this.mCtx[c].fillText("0",
                            this.offsetW - this.mCtx[c].measureText("0")
                              .width / 2 + this.miniW / 2,
                            this.offsetH + this.miniH + 2);
      this.mCtx[c].fillText(-this.data.xSize / 4,
                            this.offsetW - this.mCtx[c]
                              .measureText(-this.data.xSize / 4).width / 2 +
                              this.miniW / 4, this.offsetH + this.miniH + 2);
      this.mCtx[c].fillText(-this.data.xSize / 2,
                            this.offsetW - this.mCtx[c]
                              .measureText(-this.data.xSize / 2).width / 2,
                            this.offsetH + this.miniH + 2);
      this.mCtx[c].textBaseline = "bottom";
      this.mCtx[c].fillText("x (mm)",
                            this.miniCW / 2 - this.mCtx[c].measureText("x(mm)")
                              .width / 2,
                            this.miniCH);
      this.mCtx[c].save();
      this.mCtx[c].rotate(-Math.PI / 2);
      this.mCtx[c].translate(-this.miniCW, 0);
      this.mCtx[c].textBaseline = "top";
      this.mCtx[c].fillText("y (mm)",
                            this.miniCW / 2 - this.mCtx[c].measureText("y(mm)")
                              .width / 2,
                            0);
      this.mCtx[c].restore();
    };


    if (data.dataType == "binned")
      this.drawMiniLegend();

  }
};
