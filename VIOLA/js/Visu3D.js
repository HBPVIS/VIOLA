if (Visu === undefined)
  var Visu = {};

Visu.Renderer3D = function(panel, data) {

  //---------------------
  //Current visualisation
  //---------------------
  this.visu = "layers";

  //--------------------
  //Timers for rendering
  //--------------------
  this.lastUpdate = new Date()
    .getTime();

  //
  //Objects for 3D timeline
  //
  this.timelines = [];

  //---------------
  //Free fly camera
  //---------------

  this.cameraRotation = Math.PI / 2;
  this.cameraTranslation = 200;
  this.cameraZoomSpeed = 20;
  this.upVector = new THREE.Vector3(0, 1, 0);
  this.rightVector = new THREE.Vector3(1, 0, 0);
  this.forwardVector = new THREE.Vector3(0, 0, 1);
  this.backwardVector = new THREE.Vector3(0, 0, -1);

  this.freeFly = false;
  this.azerty = false;

  this.time = 0;

  this.useOpacity = false;
  this.useSize = true;
  this.usePosition = false;
  this.hideInactiveBins = true;

  this.rendererW = 1000;
  this.rendererH = 625;

  this.data = data;
  if (data.dataType == "binned") {
    this.xSize = data.xNeurons;
    this.ySize = data.yNeurons;
    this.boxSize = 10;
  } else if (data.dataType == "neuron") {
    this.density = 100;
    this.mmToPixel = 100;
    this.xSize = data.xSize * this.mmToPixel;
    this.ySize = data.ySize * this.mmToPixel;
    this.boxSize = 8;
    this.geomCreated = [];
  };


  this.zTimeSize = this.data.zTimeSize;

  this.logPos = false;
  this.logSize = false;
  this.logOp = false;

  this.xMin = 0;
  this.xMax = this.xSize;
  this.yMin = 0;
  this.yMax = this.ySize;

  this.layerDisplacement = 4; //Times boxSize
  this.groupDisplacement = 1;
  this.sumScale = 10;
  this.backgroundScale = 1;
  this.lfpSpacing = 20;
  if (this.data.dataType == "binned") {
    this.lfpXSize = this.boxSize * this.xSize;
    this.lfpYSize = this.boxSize * this.ySize;
  } else if (this.data.dataType == "neuron") {
    this.lfpXSize = this.xSize;
    this.lfpYSize = this.ySize;
  };

  this.keys = {
    forward: 87
    , backward: 83
    , straf_left: 65
    , straf_right: 68
    , rot_up: 38
    , rot_down: 40
    , rot_left: 37
    , rot_right: 39
    , zoom_in: 69
    , zoom_out: 81
    , up: 82
    , down: 70
  };

  this.keyCodes = {
    a: 65,
    b: 66,
    c: 67,
    d: 68,
    e: 69,
    f: 70,
    g: 71,
    h: 72,
    i: 73,
    j: 74,
    k: 75,
    l: 76,
    m: 77,
    n: 78,
    o: 79,
    p: 80,
    q: 81,
    r: 82,
    s: 83,
    t: 84,
    u: 85,
    v: 86,
    w: 87,
    x: 88,
    y: 89,
    z: 90,
    up_arrow: 38,
    down_arrow: 40,
    right_arrow: 39,
    left_arrow: 37
  };

  this.keysDown = [];

  this.showNames = true;

  this.vDisplacement = 1;

  this.cameraDistance = 600;
  this.cameraAngleHorizontal = 50;
  this.cameraAngleVertical = 30;
  this.cameraZoom = 50;

  this.maxFov = 130;
  this.minFov = 10;

  this.displayPop = [];
  this.lPos = [];
  for (var i = 0; i < data.nLayers; i++) {
    this.lPos[i] = i;
    this.displayPop[i] = true;
  };

  this.yLayerPos = [];
  this.lNum = data.nLayers;

  this.groupLayers = true;

  this.voxels = [];
  this.oVoxels = [];
  this.lMarkers = [];
  this.sumSquares = [];
  this.oSumSquares = [];

  this.rendering = false;

  this.origin = new THREE.Vector3(0, 0, 0);

  // initialize the renderer
  this.renderer = new THREE.WebGLRenderer({ antialias: true });
  this.renderer.setClearColor(0xffffff, 1);

  this.renderer.setSize(this.rendererW, this.rendererH);

  this.panel = panel;

  panel.insertBefore(this.renderer.domElement, panel.childNodes[0]);

  var canvasOverlay = document.createElement("canvas");
  canvasOverlay.width = this.rendererW;
  canvasOverlay.height = this.rendererH;
  canvasOverlay.id = "canvasOverlay";
  canvasOverlay.style.position = "absolute";
  panel.insertBefore(canvasOverlay, panel.childNodes[0]);

  this.overlay = canvasOverlay.getContext("2d");

  panel.ondragstart = function(e) {
    e.dataTransfer.setData("elemT", "dragPanel");
    e.dataTransfer.setData("id", e.target.id);
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

  // initialize the scene
  this.layersScene = new THREE.Object3D();
  this.timelineScene = new THREE.Object3D();
  this.timelineScene.visible = false;
  this.scene = new THREE.Scene();
  this.scene.add(this.layersScene);
  this.scene.add(this.timelineScene);

  // initialize the camera which is then put over the scene
  this.perspCamera = new THREE.PerspectiveCamera(70,
                                                 this.rendererW / this.rendererH,
                                                 1,
                                                 10000);
  this.orthoCamera = new THREE.OrthographicCamera(this.rendererW / -2,
                                                  this.rendererW / 2,
                                                  this.rendererH / 2,
                                                  this.rendererH / -2,
                                                  1,
                                                  10000);

  this.camera = this.perspCamera;
  this.cameraName = "persp";
  this.scene.add(this.camera);

  var material = [new THREE.LineBasicMaterial({ color: 0xff0000,
                                                transparent: true }),
                  new THREE.LineBasicMaterial({ color: 0x00ff00,
                                                transparent: true }),
                  new THREE.LineBasicMaterial({ color: 0x0000ff,
                                                transparent: true })];

  //Axis
  var geometry = [new THREE.Geometry(),
                  new THREE.Geometry(),
                  new THREE.Geometry()];
  geometry[0].vertices.push(new THREE.Vector3(0, 0, 0),
                            new THREE.Vector3(100, 0, 0));
  geometry[1].vertices.push(new THREE.Vector3(0, 0, 0),
                            new THREE.Vector3(0, 100, 0));
  geometry[2].vertices.push(new THREE.Vector3(0, 0, 0),
                            new THREE.Vector3(0, 0, 100));

  this.axes = new THREE.Object3D();
  this.axes.add(new THREE.Line(geometry[0], material[0]));
  this.axes.add(new THREE.Line(geometry[1], material[1]));
  this.axes.add(new THREE.Line(geometry[2], material[2]));

  var markerG = new THREE.Geometry();
  if (this.data.dataType == "binned") {
    markerG.vertices.push(
      new THREE.Vector3(-this.xSize / 2 * this.boxSize,
                        0,
                        -this.ySize / 2 * this.boxSize),
      new THREE.Vector3(-this.xSize / 2 * this.boxSize,
                        0,
                        this.ySize / 2 * this.boxSize),
      new THREE.Vector3(this.xSize / 2 * this.boxSize,
                        0,
                        this.ySize / 2 * this.boxSize),
      new THREE.Vector3(this.xSize / 2 * this.boxSize,
                        0,
                        -this.ySize / 2 * this.boxSize),
      new THREE.Vector3(-this.xSize / 2 * this.boxSize,
                        0,
                        -this.ySize / 2 * this.boxSize));
  } else if (this.data.dataType == "neuron") {
    markerG.vertices.push(
      new THREE.Vector3(-this.xSize / 2 - this.boxSize / 2,
                        0,
                        -this.ySize / 2 - this.boxSize / 2),
      new THREE.Vector3(-this.xSize / 2 - this.boxSize / 2,
                        0,
                        this.ySize / 2 + this.boxSize / 2),
      new THREE.Vector3(this.xSize / 2 + this.boxSize / 2,
                        0,
                        this.ySize / 2 + this.boxSize / 2),
      new THREE.Vector3(this.xSize / 2 + this.boxSize / 2,
                        0,
                        -this.ySize / 2 - this.boxSize / 2),
      new THREE.Vector3(-this.xSize / 2 - this.boxSize / 2,
                        0,
                        -this.ySize / 2 - this.boxSize / 2));
  };

  var boxGeo = new THREE.BufferGeometry()
    .fromGeometry(new THREE.BoxGeometry(this.boxSize,
                                        this.boxSize,
                                        this.boxSize));
  var squareGeo = new THREE.PlaneBufferGeometry(this.boxSize, this.boxSize);

  var voxindex;
  var i, j, k;
  var scale = 100 / (this.data.xNeurons - 2) * (this.data.xNeurons);
  var semibox = this.boxSize / 2;
  var xsemibox = this.boxSize * this.xSize / 2;
  var ysemibox = this.boxSize * this.ySize / 2;
  var xsumpos = -this.xSize * (this.boxSize) - 90;
  var ysumpos = -this.ySize * (this.boxSize) - 90;

  for (k = 0; k < this.data.nLayers; k++) {
    // 3D Timeline objects
    this.timelines[k] = new THREE.MarchingCubes(this.data.xNeurons,
                                                this.data.yNeurons,
                                                this.zTimeSize,
      new THREE.MeshPhongMaterial({ color: data.layerColors[k],
                                    specular: 0x606060,
                                    transparent: true,
                                    blending: THREE.NormalBlending }),
                                                true,
                                                true);
    this.timelineScene.add(this.timelines[k]);
    //105 ~= 100/0.95 where 0.95 ~= (40-2)/40
    this.timelines[k].scale.set(scale, scale, scale);

    // show first population upon setup
    if (k == 0) {
      this.timelines[k].visible = true;
    } else {
      this.timelines[k].visible = false;
    };

    // Layer scene objects
    if (this.data.dataType == "binned") {
      this.voxels[k] = [];
    } else if (this.data.dataType == "neuron") {
      this.geomCreated[k] = false;
    };

    this.oVoxels[k] = new THREE.Object3D();
    this.oVoxels[k].position.x = 0;
    this.oVoxels[k].position.y = this.yLayerPos[k];
    this.oVoxels[k].position.z = 0;

    this.yLayerPos[k] = (this.lNum - 1) * this.boxSize *
          this.layerDisplacement / 2 - this.lPos[k] * this.boxSize *
          this.layerDisplacement;

    if (this.data.dataType == "binned") {
      for (i = 0; i < this.xSize; i++) {
        for (j = 0; j < this.ySize; j++) {
          voxindex = j + i * this.ySize;
          this.voxels[k][voxindex] = new THREE.Mesh(boxGeo,
            new THREE.MeshLambertMaterial({ color: this.data.layerColors[k],
                                            opacity: 1,
                                            transparent: false }));
          this.voxels[k][voxindex].position.x = semibox - xsemibox +
                i * this.boxSize;
          this.voxels[k][voxindex].position.y = 0;
          this.voxels[k][voxindex].position.z = semibox - ysemibox +
                j * this.boxSize;
          this.voxels[k][voxindex].visible = false;
          this.oVoxels[k].add(this.voxels[k][voxindex]);
        };
      };

      //Sum Displays
      this.sumSquares[k] = [];
      this.oSumSquares[k] = new THREE.Object3D();

      for (i = 0; i < this.xSize; i++) {
        this.sumSquares[k][i] = new THREE.Mesh(squareGeo,
          new THREE.MeshLambertMaterial({ color: this.data.layerColors[k] }));
        this.sumSquares[k][i].position.x =
          semibox - xsemibox + i * this.boxSize;
        this.sumSquares[k][i].position.y = 0;
        this.sumSquares[k][i].position.z = ysumpos;
        this.sumSquares[k][i]
          .lookAt(new THREE.Vector3(this.sumSquares[k][i].position.x, 0, 0));
        this.oSumSquares[k].add(this.sumSquares[k][i]);
      };
      for (i = this.xSize; i < this.xSize + this.ySize; i++) {
        this.sumSquares[k][i] = new THREE.Mesh(squareGeo,
          new THREE.MeshLambertMaterial({ color: this.data.layerColors[k] }));
        this.sumSquares[k][i].position.x = xsumpos;
        this.sumSquares[k][i].position.y = 0;
        this.sumSquares[k][i].position.z =
          semibox - ysemibox + (i - this.xSize) * this.boxSize;
        this.sumSquares[k][i].lookAt(
          new THREE.Vector3(0, 0,this.sumSquares[k][i].position.z));
        this.oSumSquares[k].add(this.sumSquares[k][i]);
      };
      this.oVoxels[k].add(this.oSumSquares[k]);
    };

    this.lMarkers[k] = new THREE.Line(markerG,
      new THREE.LineBasicMaterial({ color: this.data.layerColors[k] }));
    this.oVoxels[k].add(this.lMarkers[k]);
    this.layersScene.add(this.oVoxels[k]);
  };

  //LFP
  this.lfpCanvas = document.createElement("canvas");
  this.lfpCanvas.width = this.data.lfpXSize;
  this.lfpCanvas.height = this.data.lfpYSize;
  this.lfpCtx = this.lfpCanvas.getContext("2d");
  this.lfpTexture = new THREE.Texture(this.lfpCanvas);
  this.lfpTexture.minFilter = THREE.LinearFilter;
  this.lfpTexture.magFilter = THREE.LinearFilter;

  this.lfpMaterial = new THREE.MeshBasicMaterial({ map: this.lfpTexture,
                                                   side: THREE.DoubleSide,
                                                   transparent: true,
                                                   opacity: 0.5,
                                                   depthWrite: false });
  this.lfpMesh = new THREE.Mesh(new THREE.PlaneBufferGeometry(this.lfpXSize,
                                                              this.lfpYSize),
                                this.lfpMaterial);
  this.lfpMesh.position.y =
    this.yLayerPos[this.data.nLayers - 1] - this.lfpSpacing;
  this.lfpMesh.lookAt(this.origin);
  this.lfpMesh.rotateZ(-Math.PI / 2);
  if (this.data.dataType == "binned")
    this.layersScene.add(this.lfpMesh);

  this.lfpPosition = [this.data.layerNames.indexOf("L5E"),
                      this.data.layerNames.indexOf("L5I")];
  this.lfpBottom = true;

  if (this.lfpPosition[0] != -1 && this.lfpPosition[1] != -1) {
    this.lfpBottom = false;
  }

  if (this.data.dataType == "binned") {

    //Boundaries for 3D Timeline
    this.timelineIndex = new THREE.Mesh(new THREE.PlaneBufferGeometry(2, 2), //new THREE.MeshLambertMaterial({color:0x404040, transparent:true, opacity:0.5, side:THREE.DoubleSide}));
      //this.lfpMaterial);
      new THREE.MeshBasicMaterial({ map: this.lfpTexture, side: THREE.DoubleSide, transparent: false, opacity: 0.5 }));
    this.timelineIndex.scale.set(2.5 * this.ySize, 2.5 * this.xSize, 1);
    this.timelineIndex.lookAt(this.forwardVector);
    this.timelineIndex.rotateZ(Math.PI / 2);
    this.timelineScene.add(this.timelineIndex);

    this.timelineBoundingBox = new THREE.Mesh(new THREE.BufferGeometry()
      .fromGeometry(new THREE.BoxGeometry(1, 1, 1))
      , new THREE.MeshLambertMaterial({ color: 0x808080, transparent: true, opacity: 0.2, depthWrite: false }));
    // The reference size for the timeline is 40, so we put 38 = 40 - 2
    //this.timelineBoundingBox.scale.set(5 * this.xSize, 5 * this.ySize, 200 * (this.zTimeSize - 2) / 38);

    this.timelineScene.add(this.timelineBoundingBox);

    //Scales for timeline
    var geo = new THREE.Geometry();
    geo.vertices.push(new THREE.Vector3(-1, 0, 0),
                      new THREE.Vector3(1, 0, 0),
                      new THREE.Vector3(-1, 0, -0.02),
                      new THREE.Vector3(-1, 0, 0.02),
                      new THREE.Vector3(0, 0, -0.02),
                      new THREE.Vector3(0, 0, 0.02),
                      new THREE.Vector3(1, 0, -0.02),
                      new THREE.Vector3(1, 0, 0.02));
    var linematerial = new THREE.LineBasicMaterial({ color: 0x000000,
                                                     lineWidth: 4,
                                                     depthTest: false });
    this.timelineScaleX = new THREE.Line(geo, linematerial, THREE.LinePieces);
    this.timelineScaleX.position.x = 0;
    this.timelineScaleX.position.y = -2.5 * this.ySize;
    this.timelineScaleX.position.z = -100 * (this.zTimeSize - 2) /
      (this.data.xNeurons - 2);
    this.timelineScaleX.scale.set(2.5 * this.xSize, 200, 200);
    this.timelineScene.add(this.timelineScaleX);
    this.timelineScaleY = new THREE.Line(geo, linematerial, THREE.LinePieces);
    this.timelineScaleY.position.x = -2.5 * this.xSize;
    this.timelineScaleY.position.y = 0;
    this.timelineScaleY.position.z = -100 * (this.zTimeSize - 2) /
      (this.data.xNeurons - 2);
    this.timelineScaleY.scale.set(2.5 * this.ySize, 200, 200);
    this.timelineScaleY.rotateZ(Math.PI / 2);
    this.timelineScene.add(this.timelineScaleY);
    var geo = new THREE.Geometry();
    geo.vertices.push(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, -1)
      , new THREE.Vector3(-0.04, 0, 0), new THREE.Vector3(0.04, 0, 0)
      , new THREE.Vector3(-0.04, 0, 0.4), new THREE.Vector3(0.04, 0, 0.4)
      , new THREE.Vector3(-0.04, 0, 0.8), new THREE.Vector3(0.04, 0, 0.8)
      , new THREE.Vector3(-0.04, 0, -0.4), new THREE.Vector3(0.04, 0, -0.4)
      , new THREE.Vector3(-0.04, 0, -0.8), new THREE.Vector3(0.04, 0, -0.8));
    this.timelineScaleT = new THREE.Line(geo, linematerial, THREE.LinePieces);
    this.timelineScaleT.position.x = 2.5 * this.xSize;
    this.timelineScaleT.position.y = -2.5 * this.ySize;
    // The reference size for the timeline is 40, so we put 38 = 40 - 2
    this.timelineScaleT.scale.set(100, 100, 100 * (this.zTimeSize - 2) / 38);
    this.timelineScene.add(this.timelineScaleT);


    this.zeroCanvas = document.createElement("canvas");
    this.zeroCanvas.width = 20;
    this.zeroCanvas.height = 20;
    this.zeroctx = this.zeroCanvas.getContext("2d");
    this.zeroTexture = new THREE.Texture(this.zeroCanvas);
    this.zeroTexture.minFilter = THREE.NearestFilter;
    this.zeroMaterial = new THREE.SpriteMaterial({ map: this.zeroTexture,
                                                   depthTest: false });
    this.zeroctx.font = "20px serif";
    this.zeroctx.fillStyle = "black";
    this.zeroctx.textBaseline = "middle";
    this.zeroctx.fillText("0", 10 - this.zeroctx.measureText("0")
      .width / 2, 10);
    this.zeroTexture.needsUpdate = true;

    this.maxXCanvas = document.createElement("canvas");
    this.maxXCanvas.width = 20;
    this.maxXCanvas.height = 20;
    this.maxXctx = this.maxXCanvas.getContext("2d");
    this.maxXTexture = new THREE.Texture(this.maxXCanvas);
    this.maxXTexture.minFilter = THREE.NearestFilter;
    this.maxXMaterial = new THREE.SpriteMaterial({ map: this.maxXTexture,
                                                   depthTest: false });
    this.maxXctx.font = "20px sans-serif";
    this.maxXctx.fillStyle = "black";
    this.maxXctx.textBaseline = "middle";
    this.maxXctx.fillText(this.data.xSize / 2,
                          10 - this.maxXctx.measureText(this.data.xSize / 2)
                            .width / 2, 10);
    this.maxXTexture.needsUpdate = true;

    this.minXCanvas = document.createElement("canvas");
    this.minXCanvas.width = 20;
    this.minXCanvas.height = 20;
    this.minXctx = this.minXCanvas.getContext("2d");
    this.minXTexture = new THREE.Texture(this.minXCanvas);
    this.minXTexture.minFilter = THREE.NearestFilter;
    this.minXMaterial = new THREE.SpriteMaterial({ map: this.minXTexture,
                                                   depthTest: false });
    this.minXctx.font = "20px sans-serif";
    this.minXctx.fillStyle = "black";
    this.minXctx.textBaseline = "middle";
    this.minXctx.fillText(-this.data.xSize / 2,
                          10 - this.minXctx.measureText(-this.data.xSize / 2)
                            .width / 2, 10);
    this.minXTexture.needsUpdate = true;

    this.maxYCanvas = document.createElement("canvas");
    this.maxYCanvas.width = 20;
    this.maxYCanvas.height = 20;
    this.maxYctx = this.maxYCanvas.getContext("2d");
    this.maxYTexture = new THREE.Texture(this.maxYCanvas);
    this.maxYTexture.minFilter = THREE.NearestFilter;
    this.maxYMaterial = new THREE.SpriteMaterial({ map: this.maxYTexture,
                                                   depthTest: false });
    this.maxYctx.font = "20px sans-serif";
    this.maxYctx.fillStyle = "black";
    this.maxYctx.textBaseline = "middle";
    this.maxYctx.fillText(this.data.ySize / 2,
                          10 - this.maxYctx.measureText(this.data.ySize / 2)
                            .width / 2, 10);
    this.maxYTexture.needsUpdate = true;

    this.minYCanvas = document.createElement("canvas");
    this.minYCanvas.width = 20;
    this.minYCanvas.height = 20;
    this.minYctx = this.minYCanvas.getContext("2d");
    this.minYTexture = new THREE.Texture(this.minYCanvas);
    this.minYTexture.minFilter = THREE.NearestFilter;
    this.minYMaterial = new THREE.SpriteMaterial({ map: this.minYTexture,
                                                   depthTest: false });
    this.minYctx.font = "20px sans-serif";
    this.minYctx.fillStyle = "black";
    this.minYctx.textBaseline = "middle";
    this.minYctx.fillText(-this.data.ySize / 2,
                          10 - this.minYctx.measureText(-this.data.ySize / 2)
                            .width / 2, 10);
    this.minYTexture.needsUpdate = true;

    this.xCanvas = document.createElement("canvas");
    this.xCanvas.width = 60;
    this.xCanvas.height = 60;
    this.xctx = this.xCanvas.getContext("2d");
    this.xTexture = new THREE.Texture(this.xCanvas);
    this.xTexture.minFilter = THREE.NearestFilter;
    this.xMaterial = new THREE.SpriteMaterial({ map: this.xTexture,
                                                depthTest: false });
    this.xctx.font = "20px sans-serif";
    this.xctx.fillStyle = "black";
    this.xctx.textBaseline = "middle";
    this.xctx.fillText("x(mm)", 30 - this.xctx.measureText("x(mm)")
                         .width / 2, 30);
    this.xTexture.needsUpdate = true;

    this.yCanvas = document.createElement("canvas");
    this.yCanvas.width = 60;
    this.yCanvas.height = 60;
    this.yctx = this.yCanvas.getContext("2d");
    this.yTexture = new THREE.Texture(this.yCanvas);
    this.yTexture.minFilter = THREE.NearestFilter;
    this.yMaterial = new THREE.SpriteMaterial({ map: this.yTexture,
                                                depthTest: false,
                                                rotation: Math.PI / 2 });
    this.yctx.font = "20px sans-serif";
    this.yctx.fillStyle = "black";
    this.yctx.textBaseline = "middle";
    this.yctx.fillText("y(mm)", 30 - this.yctx.measureText("y(mm)")
      .width / 2, 30);
    this.yTexture.needsUpdate = true;

    this.tCanvas = document.createElement("canvas");
    this.tCanvas.width = 100;
    this.tCanvas.height = 100;
    this.tctx = this.tCanvas.getContext("2d");
    this.tTexture = new THREE.Texture(this.tCanvas);
    this.tTexture.minFilter = THREE.NearestFilter;
    this.tMaterial = new THREE.SpriteMaterial({ map: this.tTexture,
                                                depthTest: false });
    this.tctx.font = "20px sans-serif";
    this.tctx.fillStyle = "black";
    this.tctx.textBaseline = "middle";
    this.tctx.fillText("Delay (ms)", 50 - this.tctx.measureText("Delay (ms)")
                         .width / 2, 50);
    this.tTexture.needsUpdate = true;

    this.tm40Canvas = document.createElement("canvas");
    this.tm40Canvas.width = 60;
    this.tm40Canvas.height = 60;
    this.tm40ctx = this.tm40Canvas.getContext("2d");
    this.tm40Texture = new THREE.Texture(this.tm40Canvas);
    this.tm40Texture.minFilter = THREE.NearestFilter;
    this.tm40Material = new THREE.SpriteMaterial({ map: this.tm40Texture,
                                                   depthTest: false });
    this.tm40ctx.font = "20px sans-serif";
    this.tm40ctx.fillStyle = "black";
    this.tm40ctx.textBaseline = "middle";
    this.tm40ctx.fillText(Math.round(-this.zTimeSize * this.data.resolution *
                                     0.2),
                          30 - this.tm40ctx.measureText(Math.round(
                            -this.zTimeSize * this.data.resolution * 0.2))
                            .width / 2,
                          30);
    this.tm40Texture.needsUpdate = true;

    this.tm80Canvas = document.createElement("canvas");
    this.tm80Canvas.width = 60;
    this.tm80Canvas.height = 60;
    this.tm80ctx = this.tm80Canvas.getContext("2d");
    this.tm80Texture = new THREE.Texture(this.tm80Canvas);
    this.tm80Texture.minFilter = THREE.NearestFilter;
    this.tm80Material = new THREE.SpriteMaterial({ map: this.tm80Texture,
                                                   depthTest: false });
    this.tm80ctx.font = "20px sans-serif";
    this.tm80ctx.fillStyle = "black";
    this.tm80ctx.textBaseline = "middle";
    this.tm80ctx.fillText(Math.round(-this.zTimeSize * this.data.resolution *
                                       0.4),
                          30 - this.tm80ctx.measureText(
                            Math.round(-this.zTimeSize * this.data.resolution *
                                       0.4)).width / 2,
                          30);
    this.tm80Texture.needsUpdate = true;

    this.t40Canvas = document.createElement("canvas");
    this.t40Canvas.width = 60;
    this.t40Canvas.height = 60;
    this.t40ctx = this.t40Canvas.getContext("2d");
    this.t40Texture = new THREE.Texture(this.t40Canvas);
    this.t40Texture.minFilter = THREE.NearestFilter;
    this.t40Material = new THREE.SpriteMaterial({ map: this.t40Texture,
                                                  depthTest: false });
    this.t40ctx.font = "20px sans-serif";
    this.t40ctx.fillStyle = "black";
    this.t40ctx.textBaseline = "middle";
    this.t40ctx.fillText(Math.round(this.zTimeSize * this.data.resolution *
                                      0.2),
                         30 - this.t40ctx.measureText(
                           Math.round(this.zTimeSize * this.data.resolution *
                                        0.2)).width / 2,
                         30);
    this.t40Texture.needsUpdate = true;

    this.t80Canvas = document.createElement("canvas");
    this.t80Canvas.width = 60;
    this.t80Canvas.height = 60;
    this.t80ctx = this.t80Canvas.getContext("2d");
    this.t80Texture = new THREE.Texture(this.t80Canvas);
    this.t80Texture.minFilter = THREE.NearestFilter;
    this.t80Material = new THREE.SpriteMaterial({ map: this.t80Texture,
                                                  depthTest: false });
    this.t80ctx.font = "20px sans-serif";
    this.t80ctx.fillStyle = "black";
    this.t80ctx.textBaseline = "middle";
    this.t80ctx.fillText(Math.round(this.zTimeSize * this.data.resolution *
                                      0.4),
                         30 - this.t80ctx.measureText(
                           Math.round(this.zTimeSize * this.data.resolution *
                                      0.4)).width / 2,
                         30);
    this.t80Texture.needsUpdate = true;

    //X scale
    this.xzeroSprite = new THREE.Sprite(this.zeroMaterial);
    this.xzeroSprite.scale.set(20, 20, 20);
    this.xzeroSprite.position.x = 0;
    this.xzeroSprite.position.y = -2.5 * this.ySize;
    this.xzeroSprite.position.z = -100 * (this.zTimeSize - 2) /
      (this.data.xNeurons - 2) - 20;
    this.timelineScene.add(this.xzeroSprite);
    this.xMinSprite = new THREE.Sprite(this.minXMaterial);
    this.xMinSprite.scale.set(20, 20, 20);
    this.xMinSprite.position.x = -2.5 * this.xSize + 10;
    this.xMinSprite.position.y = -2.5 * this.ySize;
    this.xMinSprite.position.z = -100 * (this.zTimeSize - 2) /
      (this.data.xNeurons - 2) - 20;
    this.timelineScene.add(this.xMinSprite);
    this.xMaxSprite = new THREE.Sprite(this.maxXMaterial);
    this.xMaxSprite.scale.set(20, 20, 20);
    this.xMaxSprite.position.x = 2.5 * this.xSize - 10;
    this.xMaxSprite.position.y = -2.5 * this.ySize;
    this.xMaxSprite.position.z = -100 * (this.zTimeSize - 2) /
      (this.data.xNeurons - 2) - 20;
    this.timelineScene.add(this.xMaxSprite);
    this.xSprite = new THREE.Sprite(this.xMaterial);
    this.xSprite.scale.set(60, 60, 60);
    this.xSprite.position.x = 0;
    this.xSprite.position.y = -2.5 * this.ySize - 20;
    this.xSprite.position.z = -100 * (this.zTimeSize - 2) /
      (this.data.xNeurons - 2) - 45;
    this.timelineScene.add(this.xSprite);

    //Y scale
    this.yzeroSprite = new THREE.Sprite(this.zeroMaterial);
    this.yzeroSprite.scale.set(20, 20, 20);
    this.yzeroSprite.position.x = -2.5 * this.xSize;
    this.yzeroSprite.position.y = 0;
    this.yzeroSprite.position.z = -100 * (this.zTimeSize - 2) /
      (this.data.xNeurons - 2) - 20;
    this.timelineScene.add(this.yzeroSprite);
    this.yMinSprite = new THREE.Sprite(this.minYMaterial);
    this.yMinSprite.scale.set(20, 20, 20);
    this.yMinSprite.position.x = -2.5 * this.xSize;
    this.yMinSprite.position.y = -2.5 * this.ySize + 10;
    this.yMinSprite.position.z = -100 * (this.zTimeSize - 2) /
      (this.data.xNeurons - 2) - 20;
    this.timelineScene.add(this.yMinSprite);
    this.yMaxSprite = new THREE.Sprite(this.maxYMaterial);
    this.yMaxSprite.scale.set(20, 20, 20);
    this.yMaxSprite.position.x = -2.5 * this.xSize;
    this.yMaxSprite.position.y = 2.5 * this.ySize - 10;
    this.yMaxSprite.position.z = -100 * (this.zTimeSize - 2) /
      (this.data.xNeurons - 2) - 20;
    this.timelineScene.add(this.yMaxSprite);
    this.ySprite = new THREE.Sprite(this.yMaterial);
    this.ySprite.scale.set(60, 60, 60);
    this.ySprite.position.x = -2.5 * this.xSize;
    this.ySprite.position.y = 0;
    this.ySprite.position.z = -100 * (this.zTimeSize - 2) /
      (this.data.xNeurons - 2) - 45;
    this.timelineScene.add(this.ySprite);

    //Time scale
    this.tSprite = new THREE.Sprite(this.tMaterial);
    this.tSprite.scale.set(100, 100, 100);
    this.tSprite.position.x = 2.5 * this.xSize + 20;
    this.tSprite.position.y = -2.5 * this.ySize - 20;
    this.timelineScene.add(this.tSprite);
    this.tm40Sprite = new THREE.Sprite(this.tm40Material);
    this.tm40Sprite.scale.set(60, 60, 60);
    this.tm40Sprite.position.x = 2.5 * this.xSize + 20;
    this.tm40Sprite.position.y = -2.5 * this.ySize;
    this.tm40Sprite.position.z = 100 * (this.zTimeSize - 2) /
      (this.data.xNeurons - 2) * 0.4;
    this.timelineScene.add(this.tm40Sprite);
    this.tm80Sprite = new THREE.Sprite(this.tm80Material);
    this.tm80Sprite.scale.set(60, 60, 60);
    this.tm80Sprite.position.x = 2.5 * this.xSize + 20;
    this.tm80Sprite.position.y = -2.5 * this.ySize;
    this.tm80Sprite.position.z = 100 * (this.zTimeSize - 2) /
      (this.data.xNeurons - 2) * 0.8;
    this.timelineScene.add(this.tm80Sprite);
    this.t40Sprite = new THREE.Sprite(this.t40Material);
    this.t40Sprite.scale.set(60, 60, 60);
    this.t40Sprite.position.x = 2.5 * this.xSize + 20;
    this.t40Sprite.position.y = -2.5 * this.ySize;
    this.t40Sprite.position.z = -100 * (this.zTimeSize - 2) /
      (this.data.xNeurons - 2) * 0.4;
    this.timelineScene.add(this.t40Sprite);
    this.t80Sprite = new THREE.Sprite(this.t80Material);
    this.t80Sprite.scale.set(60, 60, 60);
    this.t80Sprite.position.x = 2.5 * this.xSize + 20;
    this.t80Sprite.position.y = -2.5 * this.ySize;
    this.t80Sprite.position.z = -100 * (this.zTimeSize - 2) /
      (this.data.xNeurons - 2) * 0.8;
    this.timelineScene.add(this.t80Sprite);

    //Plane for sum background
    this.basePlaneHeight =
      this.yLayerPos[0] - this.yLayerPos[this.data.nLayers - 1];

    this.plane = new THREE.Mesh(
      new THREE.PlaneBufferGeometry((2 * this.boxSize + 2) * this.ySize,
                                    this.basePlaneHeight),
      new THREE.MeshLambertMaterial({ color: 0x404040 }));
    this.plane.position.set(-this.xSize * this.boxSize - 100, 0, 0);
    this.plane.lookAt(this.origin);
    //this.layersScene.add(this.plane); //switched off by default

    this.plane2 = new THREE.Mesh(
      new THREE.PlaneBufferGeometry((2 * this.boxSize + 2) * this.xSize,
                                    this.basePlaneHeight),
      new THREE.MeshLambertMaterial({ color: 0x404040 }));
    this.plane2.position.set(0, 0, -this.ySize * this.boxSize - 100);
    this.plane2.lookAt(this.origin);
    //this.layersScene.add(this.plane2);
  }

  //Lights
  var light = new THREE.AmbientLight(0x404040); // soft white light
  this.layersScene.add(light);

  //not fixed to camera anymore
  var cameraLight = new THREE.DirectionalLight(0xffffff, 0.5);
  cameraLight.position.set(0, -1, 0);
  var directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.5);
  directionalLight1.position.set(1, 1, -1);
  var directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
  directionalLight2.position.set(-1, -1, 1);
  var directionalLight3 = new THREE.DirectionalLight(0xffffff, 0.5);
  directionalLight3.position.set(0, 1, 0);
  this.layersScene.add(cameraLight);
  this.layersScene.add(directionalLight1);
  this.layersScene.add(directionalLight2);
  this.layersScene.add(directionalLight3);


  var light = new THREE.AmbientLight(0x404040); // soft white light
  this.timelineScene.add(light);

  //not fixed to camera anymore
  var cameraLight = new THREE.DirectionalLight(0xffffff, 0.5);
  cameraLight.position.set(0, -1, 0);
  var directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.5);
  directionalLight1.position.set(1, 1, -1);
  var directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
  directionalLight2.position.set(-1, -1, 1);
  var directionalLight3 = new THREE.DirectionalLight(0xffffff, 0.5);
  directionalLight3.position.set(0, 1, 0);

  this.timelineScene.add(cameraLight);
  this.timelineScene.add(directionalLight1);
  this.timelineScene.add(directionalLight2);
  this.timelineScene.add(directionalLight3);


  this.camera.position.x = Math.floor(
    Math.cos(this.cameraAngleHorizontal * Math.PI / 180) *
    Math.cos(this.cameraAngleVertical * Math.PI / 180) * this.cameraDistance);
  this.camera.position.z = Math.floor(
    Math.sin(this.cameraAngleHorizontal *Math.PI / 180) *
    Math.cos(this.cameraAngleVertical * Math.PI / 180) * this.cameraDistance);
  this.camera.position.y = Math.floor(
    Math.sin(this.cameraAngleVertical * Math.PI / 180) * this.cameraDistance);

  this.camera.lookAt(this.origin);

  this.cameraDirection = new THREE.Vector3(-this.camera.position.x,
                                           -this.camera.position.y,
                                           -this.camera.position.z)
    .normalize();
  this.cameraLateralDirection = new THREE.Vector3(-this.camera.position.x,
                                                  -this.camera.position.y,
                                                  -this.camera.position.z)
    .applyAxisAngle(new THREE.Vector3(0, -1, 0), Math.PI / 2)
    .normalize();

  this.keyEventHandler = this.moveCameraFixed;

  this.layerMenuItems = [];
  this.layerMenuItems[0] = document.createElement('li');
  this.layerMenuItems[0].className = "subMenuItem";
  this.layerMenuItems[0].innerHTML = '<div>Spacing ></div>' +
    '<ul style="width:131px">' +
    '<li>' +
    '<div>Global <input type="number" value="4" style="width:40px"' +
      'onchange="r3D.setLayerSpacing(this.value)"></div>' +
    '</li>' +
    '<li>' +
    '<div>LFP <input type="number" value="2" style="width:40px"' +
      'onchange="r3D.setLFPSpacing(this.value)"></div>' +
    '</li>' +
    '<li class="spacer"></li>' +
    '<li>' +
    '<div>Group <input id="spacingInput" type="number" value="1"' +
      'style="width:40px" onchange="r3D.setGroupSpacing(this.value)"></div>' +
    '</li>' +
    '<li>' +
    '<div><input type="checkbox" checked' +
      'onchange="r3D.toggleLayerGrouping(this)">Group E-I Layers</div>' +
    '</li>' +
    '</ul>';
  document.getElementById('3DViewSubMenu')
    .appendChild(this.layerMenuItems[0]);

  this.layerMenuItems[1] = document.createElement('li');
  this.layerMenuItems[1].className = "subMenuItem";
  this.layerMenuItems[1].innerHTML = '<div>Spikes Display ></div>' +
    '<ul style="width:200px">' +
    '<li>' +
    '<div><input type="checkbox"' +
      'onchange="r3D.toggleOpacity(this.checked)">Opacity</div>' +
    '</li>' +
    '<li class="spacer"></li>' +
    '<li>' +
    '<div><input type="checkbox" checked' +
      'onchange="r3D.toggleSize(this.checked)">Size</div>' +
    '</li>' +
    '<li>' +
    '<div><input type="checkbox" disabled id="hideCheck" checked' +
      'onchange="r3D.toggleHideNeurons(this.checked)">Hide inactive bins' +
        '</div>' +
    '</li>' +
    '<li class="spacer"></li>' +
    '<li>' +
    '<div><input type="checkbox"' +
      'onchange="r3D.togglePosition(this.checked)">Position</div>' +
    '</li>' +
    '<li>' +
    '<div><input id="displacementInput" disabled type="number" value="1"' +
      'style="width:40px" onchange="r3D.setVerticalDisplacement(this.value)">' +
      'Vertical displacement</div>' +
    '</li>' +
    '</ul>';
  document.getElementById('3DViewSubMenu')
    .appendChild(this.layerMenuItems[1]);

  this.layerMenuItems[2] = document.createElement('li');
  this.layerMenuItems[2].className = "subMenuItem";
  this.layerMenuItems[2].innerHTML = '<div>LFP ></div>' +
    '<ul style="width:200px">' +
    '<li>' +
    '<div>Opacity <input type="checkbox" checked' +
      'onchange="r3D.toggleLFPOpacity()"><input id="lfpopacityslider"' +
      'type="range" min="0" max="10" step="1" value="5"' +
      'oninput="r3D.setLFPOpacity(this.value)"' +
      'style="width:100px;vertical-align:middle;display:inline"></div>' +
    '</li>' +
    '</ul>';
  document.getElementById('3DViewSubMenu')
    .appendChild(this.layerMenuItems[2]);

  this.timelineMenuItems = [];
  this.timelineMenuItems[0] = document.createElement('li');

  this.timeWindowValue = document.getElementById('timeWindowValue');
  this.timeWindowRange = document.getElementById('timeWindowRange');

  this.timelineOffset = 0;

  this.updateLayerPositions();
};

Visu.Renderer3D.prototype = {

  setIsolation: function(v) {
    var val = parseInt(v);
    for (var i = 0; i < this.data.nLayers; i++) {
      this.timelines[i].setIsolation(val);
    }
    updated = true;
  },

  setDelay: function(v, i) {
    var val = parseInt(v);
    this.timelines[i].setDelay(val);
    document.getElementById('delayspan' + i)
      .innerHTML = val;
    updated = true;
  },

  setOpacity: function(v, i) {
    var val = parseInt(v);
    this.timelines[i].material.opacity = val / 10;
    if (val != 10) {
      this.timelines[i].material.depthWrite = false;
    } else {
      this.timelines[i].material.depthWrite = true;
    };

    updated = true;
  },

  setTimeWindow: function(v, w, index) {
    var val = parseInt(v);
    this.zTimeSize = val;

    // The reference size for the timeline is 40, so we put 38 = 40 - 2
    var zscale = (this.zTimeSize - 2) / 38;

    this.timelineBoundingBox.scale.set(5 * this.xSize,
                                       5 * this.ySize,
                                       200 * zscale);
    this.timelineScaleT.scale.set(100, 100, 100 * zscale);
    this.tm40ctx.clearRect(0, 0, 60, 60);
    this.tm40ctx.fillText(-Math.round(this.zTimeSize * this.data.resolution *
                                        0.2),
                          30 - this.t40ctx.measureText(
                            -Math.round(this.zTimeSize * this.data.resolution *
                                          0.2)).width / 2,
                          30);
    this.tm40Texture.needsUpdate = true;
    this.tm40Sprite.position.z = 100 * zscale * 0.4;
    this.tm80ctx.clearRect(0, 0, 60, 60);
    this.tm80ctx.fillText(-Math.round(this.zTimeSize * this.data.resolution *
                                        0.4),
                          30 - this.t80ctx.measureText(
                            -Math.round(this.zTimeSize * this.data.resolution *
                                          0.4)).width / 2,
                          30);
    this.tm80Texture.needsUpdate = true;
    this.tm80Sprite.position.z = 100 * zscale * 0.8;
    this.t40ctx.clearRect(0, 0, 60, 60);
    this.t40ctx.fillText(Math.round(this.zTimeSize * this.data.resolution *
                                    0.2),
                         30 - this.t40ctx.measureText(
                           Math.round(this.zTimeSize * this.data.resolution *
                                        0.2)).width / 2,
                         30);
    this.t40Texture.needsUpdate = true;
    this.t40Sprite.position.z = -100 * zscale * 0.4;
    this.t80ctx.clearRect(0, 0, 60, 60);
    this.t80ctx.fillText(Math.round(this.zTimeSize * this.data.resolution *
                                    0.4),
                         30 - this.t80ctx.measureText(
                           Math.round(this.zTimeSize * this.data.resolution *
                                        0.4)).width / 2,
                         30);
    this.t80Texture.needsUpdate = true;
    this.t80Sprite.position.z = -100 * zscale * 0.8;

    this.xzeroSprite.position.z = -100 * zscale - 20;
    this.xMinSprite.position.z = -100 * zscale - 20;
    this.xMaxSprite.position.z = -100 * zscale - 20;
    this.xSprite.position.z = -100 * zscale - 45;

    this.yzeroSprite.position.z = -100 * zscale - 20;
    this.yMinSprite.position.z = -100 * zscale - 20;
    this.yMaxSprite.position.z = -100 * zscale - 20;
    this.ySprite.position.z = -100 * zscale - 45;

    this.timelineScaleX.position.z = -100 * zscale;
    this.timelineScaleY.position.z = -100 * zscale;

    if (index < this.zTimeSize / 2) {
      this.timeOffset = 0;
    } else if (index >= this.data.timestamps - this.zTimeSize / 2) {
      this.timeOffset = this.data.timestamps - this.zTimeSize;
    } else {
      this.timeOffset = Math.round(index - this.zTimeSize / 2);
    };

    if (w == 0) {
      for (var i = 0; i < this.data.nLayers; i++) {
        this.timelines[i].setSizeZ(val);
        this.timelines[i].setTimeOffset(this.timeOffset);
      };
      this.timeWindowValue.value = val;
    } else {
      for (var i = 0; i < this.data.nLayers; i++) {
        this.timelines[i].setSizeZ(val);
        this.timelines[i].setTimeOffset(this.timeOffset);
      };
      this.timeWindowRange.value = val;
    };
    updateData();
    updated = true;
  },

  setDensity: function(v) {
    this.density = parseInt(v);
    updated = true;
  },

  disableDrag: function(v) {
    if (v) {
      this.panel.draggable = false;
    } else {
      this.panel.draggable = true;
    }
  },

  setLayerSpacing: function(v) {
    this.layerDisplacement = parseInt(v);
    this.updateLayerPositions();
    updated = true;
  },

  setTime: function(v) {
    this.time = v;
    updated = true;
  },

  setVerticalDisplacement: function(v) {
    this.vDisplacement = parseInt(v);
    this.updateLayerPositions();
    updated = true;
  },

  setGroupSpacing: function(v) {
    this.groupDisplacement = parseInt(v);
    this.updateLayerPositions();
    updated = true;
  },

  setLFPSpacing: function(v) {
    this.lfpSpacing = this.boxSize * parseInt(v);
    this.updateLayerPositions();
    updated = true;
  },

  setLFPOpacity: function(v) {
    this.lfpMesh.material.opacity = v / 10;
    updated = true;
  },

  setMinX: function(v) {
    if (v >= 0 && v <= this.xSize)
      this.xMin = parseInt(v);
  },
  setMaxX: function(v) {
    if (v >= 0 && v <= this.xSize)
      this.xMax = parseInt(v);
  },
  setMinY: function(v) {
    if (v >= 0 && v <= this.ySize)
      this.yMin = parseInt(v);
  },
  setMaxY: function(v) {
    if (v >= 0 && v <= this.ySize)
      this.yMax = parseInt(v);
  },

  setSumScale: function(v) {
    this.sumScale = parseFloat(v);
    if (this.sumScale <= 0)
      this.sumScale = 1;
    updated = true;
  },

  disableDrag: function(v) {
    if (v) {
      this.panel.draggable = false;
    } else {
      this.panel.draggable = true;
    };
  },

  createNeuronGeometries: function(k) {
    if (this.displayPop[k])
      this.layersScene.remove(this.oVoxels[k]);

    if (this.geomCreated[k]) {
      this.oVoxels[k] = new THREE.Object3D();
      this.oVoxels[k].add(this.lMarkers[k]);
    }

    var nIndex;
    var boxGeo = new THREE.BufferGeometry()
      .fromGeometry(new THREE.BoxGeometry(this.boxSize,
                                          this.boxSize,
                                          this.boxSize));
    for (i = 0; i < this.data.neuronLayers[k].length; i++) {
      nIndex = this.data.neuronLayers[k][i];
      this.voxels[nIndex] = new THREE.Mesh(boxGeo,
        new THREE.MeshLambertMaterial({ color: this.data.layerColors[k],
                                        opacity: 1,
                                        transparent: false }));
      this.voxels[nIndex].position.x =
        this.data.neuronPos[nIndex][0] * this.mmToPixel;
      this.voxels[nIndex].position.y = 0;
      this.voxels[nIndex].position.z =
        this.data.neuronPos[nIndex][1] * this.mmToPixel;
      this.voxels[nIndex].visible = false;
      this.oVoxels[k].add(this.voxels[nIndex]);
    }
    this.geomCreated[k] = true;

    if (this.displayPop[k])
      this.layersScene.add(this.oVoxels[k]);

    updated = true;
  },

  adjustDisplayedNeurons: function() {
    //Remove all previous elements that will change
    for (var k = 0; k < this.data.nLayers; k++) {
      if (this.displayPop[k])
        this.layersScene.remove(this.oVoxels[k]);
      this.oVoxels[k] = new THREE.Object3D();

      //Re-add all valid neurons
      for (var i = this.xMin; i < this.xMax; i++) {
        for (var j = this.yMin; j < this.yMax; j++) {
          this.oVoxels[k].add(this.voxels[k][j + i * this.ySize]);
        }
      }

      //Re-add corresponding sumDisplays
      this.oSumSquares[k] = new THREE.Object3D();
      for (var i = this.xMin; i < this.xMax; i++)
        this.oSumSquares[k].add(this.sumSquares[k][i]);
      for (var i = this.xSize + this.yMin; i < this.xSize + this.yMax; i++)
        this.oSumSquares[k].add(this.sumSquares[k][i]);
      this.oVoxels[k].add(this.oSumSquares[k]);

      //Recreate layerMarkers
      var markerG = new THREE.Geometry();
      markerG.vertices.push(
        new THREE.Vector3((this.xMin - this.xSize / 2) * this.boxSize,
                          0,
                          (this.yMin - this.ySize / 2) * this.boxSize),
        new THREE.Vector3((this.xMin - this.xSize / 2) * this.boxSize,
                          0,
                          (this.yMax - this.ySize / 2) * this.boxSize),
        new THREE.Vector3((this.xMax - this.xSize / 2) * this.boxSize,
                          0,
                          (this.yMax - this.ySize / 2) * this.boxSize),
        new THREE.Vector3((this.xMax - this.xSize / 2) * this.boxSize,
                          0,
                          (this.yMin - this.ySize / 2) * this.boxSize),
        new THREE.Vector3((this.xMin - this.xSize / 2) * this.boxSize,
                          0,
                          (this.yMin - this.ySize / 2) * this.boxSize));
      this.lMarkers[k] =
        new THREE.Line(markerG,
          new THREE.LineBasicMaterial({ color: this.data.layerColors[k] }));
      this.oVoxels[k].add(this.lMarkers[k]);

      //Add the group again
      if (this.displayPop[k])
        this.layersScene.add(this.oVoxels[k]);
    };

    this.updateLayerPositions();
    updated = true;
  },

  toggleLFPOpacity: function() {
    if (this.lfpMesh.material.transparent == true) {
      this.lfpMesh.material.transparent = false;
      this.lfpMesh.material.depthWrite = true;
      document.getElementById('lfpopacityslider')
        .disabled = true;
    } else {
      this.lfpMesh.material.transparent = true;
      this.lfpMesh.material.depthWrite = false;
      document.getElementById('lfpopacityslider')
        .disabled = false;
    };
    updated = true;
  },

  toggleLogPos: function(b) {
    this.logPos = b;
  },

  toggleAxis: function(b) {
    if (b) {
      this.layersScene.add(this.axes);
    } else {
      this.layersScene.remove(this.axes);
    };
    updated = true;
  },

  toggleLayerGrouping: function(p) {
    this.groupLayers = p.checked;
    this.updateLayerPositions();
    updated = true;
    document.getElementById("spacingInput")
      .disabled = !p.checked;
  },

  switchScene: function() {
    if (this.visu == "layers") {
      this.toggleScene("timeline");
    } else {
      this.toggleScene("layers");
    };
  },

  toggleScene: function(v) {
    var i;
    if (v == "layers") {
      this.timelineScene.visible = false;
      for (i = 0; i < this.data.nLayers; i++) {
        this.timelines[i].reset();
      }
      this.layersScene.visible = true;
      document.getElementById("layersParams")
        .className = "";
      document.getElementById("timelineParams")
        .className = "hiddenPanel";
      document.getElementById("radioscenet")
        .checked = false;
      document.getElementById("radioscenel")
        .checked = true;

      for (i = 0; i < this.timelineMenuItems.length; i++) {
        if (this.timelineMenuItems[i].parentNode != undefined) {
          this.timelineMenuItems[i].parentNode
            .removeChild(this.timelineMenuItems[i]);
        };
      };
      for (i = 0; i < this.layerMenuItems.length; i++) {
        document.getElementById('3DViewSubMenu')
          .appendChild(this.layerMenuItems[i]);
      };
    } else {
      this.timelineScene.visible = true;
      this.layersScene.visible = false;
      document.getElementById("timelineParams")
        .className = "";
      document.getElementById("layersParams")
        .className = "hiddenPanel";
      document.getElementById("radioscenel")
        .checked = false;
      document.getElementById("radioscenet")
        .checked = true;

      for (i = 0; i < this.layerMenuItems.length; i++) {
        if (this.layerMenuItems[i].parentNode != undefined) {
          this.layerMenuItems[i].parentNode.removeChild(this.layerMenuItems[i]);
        };
      };
      for (i = 0; i < this.timelineMenuItems.length; i++) {
        document.getElementById('3DViewSubMenu')
          .appendChild(this.timelineMenuItems[i]);
      };

    };
    this.visu = v;
    updateData(); //Function defined in index.html
    updated = true;
  },

  switchCameraType: function() {
    if (this.cameraName == "persp") {
      this.toggleCamera("ortho");
    } else {
      this.toggleCamera("persp");
    };
  },

  toggleCamera: function(v) {

    this.scene.remove(this.camera);
    if (v == "persp") {
      console.log("changing to perspCamera");
      this.cameraName = "persp";
      this.camera = this.perspCamera;
    } else {
      console.log("changing to orthoCamera");
      this.cameraName = "ortho";
      this.camera = this.orthoCamera;
    };
    this.scene.add(this.camera);
    this.updateCameraPosition();
    updated = true;
  },

  togglePlane: function(b) {
    if (b) {
      this.layersScene.add(this.plane);
    } else {
      this.layersScene.remove(this.plane);
    };
    updated = true;
  },

  toggleMarkers: function(b) {
    if (b) {
      for (var i = 0; i < this.data.nLayers; i++)
        this.oVoxels[i].add(this.lMarkers[i]);
    } else {
      for (var i = 0; i < this.data.nLayers; i++)
        this.oVoxels[i].remove(this.lMarkers[i]);
    };
    updated = true;
  },

  toggleLFPs: function(b) {
    this.lfpMesh.visible = b;
    updated = true;
  },

  toggleTags: function(b) {
    this.showNames = b;
    updated = true;
  },

  toggleOpacity: function(b) {
    this.useOpacity = b;
    for (var k = 0; k < this.data.nLayers; k++) {
      for (var i = 0; i < this.xSize; i++) {
        for (var j = 0; j < this.ySize; j++) {
          this.voxels[k][j + i * this.ySize].material.transparent = b;
        };
      };
    };
  },

  toggleHideNeurons: function(b) {
    this.hideInactiveBins = b;
  },

  toggleSize: function(b) {
    var h = document.getElementById("hideCheck");
    this.useSize = b;
    if (!b) {
      for (var k = 0; k < this.data.nLayers; k++) {
        for (var i = 0; i < this.xSize; i++) {
          for (var j = 0; j < this.ySize; j++) {
            this.voxels[k][j + i * this.ySize].scale.set(1, 1, 1);
          };
        };
      };
      h.disabled = false;
    } else {
      this.hideInactiveBins = true;
      h.checked = true;
      h.disabled = true;
    };
  },

  togglePosition: function(b) {
    this.usePosition = b;
    if (!b) {
      for (var k = 0; k < data.nLayers; k++) {
        for (var i = 0; i < this.xSize; i++) {
          for (var j = 0; j < this.ySize; j++) {
            this.voxels[k][j + i * this.ySize].position.y = 0;
          };
        };
      };
    };
    document.getElementById("displacementInput").disabled = !b;
  },

  computeColor: function(d) {
    if (d < 0.33)
      return [Math.round(3 * d * 255), 0.1, 0.1];
    else if (d < 0.66)
      return [1, Math.round((d - 0.33) * 3), 0.1];
    else
      return [1, 1, Math.round((d - 0.66) * 3)];
  },

  renderScene: function() {
    this.renderer.render(this.scene, this.camera);
    this.renderOverlay();
  },

  renderOverlay: function() {
    this.overlay.clearRect(0, 0, this.rendererW, this.rendererH);
    this.overlay.font = "15px sans-serif";
    this.overlay.textBaseline = "top";
    this.overlay.fillStyle = "black";

    var mode, type;
    if (this.freeFly) {
      mode = "Free";
    } else {
      mode = "Fixed";
    }
    if (this.cameraName == "persp") {
      type = "Perspective";
    } else {
      type = "Orthographic";
    }
    this.overlay.fillText("Camera : " + type + " " + mode, 10, 10);

    this.overlay.fillText(this.time + "/" + this.data.simulationLength + " ms",
                          10,
                          30);

    // Layer names
    if (this.showNames && this.visu == "layers") {
      this.overlay.font = "30px sans-serif";
      this.overlay.textBaseline = "bottom";
      this.overlay.strokeStyle = "black";
      this.overlay.lineWidth = 1;
      for (var i = 0; i < this.data.nLayers; i++) {
        if (this.displayPop[i]) {
          this.overlay.fillStyle = this.data.layerColors[i];
          this.overlay.fillRect(0, i * this.svgH / this.data.nLayers,
                                20,
                                this.svgH / this.data.nLayers);
          this.overlay.fillText(this.data.layerNames[i],
                                10,
                                this.rendererH - 10 -
                                  (this.data.nLayers - i - 1) * 40);
          this.overlay.strokeText(this.data.layerNames[i],
                                  10,
                                  this.rendererH - 10 -
                                    (this.data.nLayers - i - 1) * 40);
        };
      };
    };
  },

  moveCameraFree: function() {
    if (this.keysDown[this.keys.forward]) {
      this.camera.translateOnAxis(this.forwardVector,
                                  -this.cameraTranslation * this.delta);
      updated = true;
    };
    if (this.keysDown[this.keys.backward]) {
      this.camera.translateOnAxis(this.forwardVector,
                                  this.cameraTranslation * this.delta);
      updated = true;
    };
    if (this.keysDown[this.keys.straf_left]) {
      this.camera.translateOnAxis(this.rightVector,
                                  -this.cameraTranslation * this.delta);
      updated = true;
    };
    if (this.keysDown[this.keys.straf_right]) {
      this.camera.translateOnAxis(this.rightVector,
                                  this.cameraTranslation * this.delta);
      updated = true;
    };
    if (this.keysDown[this.keys.rot_up]) {
      this.cameraDirection.applyAxisAngle(this.cameraLateralDirection,
                                          this.cameraRotation * this.delta);
      this.camera.lookAt(new THREE.Vector3()
        .addVectors(this.camera.position, this.cameraDirection));
      updated = true;
    };
    if (this.keysDown[this.keys.rot_down]) {
      this.cameraDirection.applyAxisAngle(this.cameraLateralDirection,
                                          -this.cameraRotation * this.delta);
      this.camera.lookAt(new THREE.Vector3()
        .addVectors(this.camera.position, this.cameraDirection));
      updated = true;
    };
    if (this.keysDown[this.keys.rot_left]) {
      this.cameraLateralDirection.applyAxisAngle(this.upVector,
                                                 this.cameraRotation *
                                                   this.delta);
      this.cameraDirection.applyAxisAngle(this.upVector,
                                          this.cameraRotation * this.delta);
      this.camera.lookAt(new THREE.Vector3()
        .addVectors(this.camera.position, this.cameraDirection));
      updated = true;
    };
    if (this.keysDown[this.keys.rot_right]) {
      this.cameraLateralDirection.applyAxisAngle(this.upVector,
                                                 -this.cameraRotation *
                                                   this.delta);
      this.cameraDirection.applyAxisAngle(this.upVector,
                                          -this.cameraRotation * this.delta);
      this.camera.lookAt(new THREE.Vector3()
        .addVectors(this.camera.position, this.cameraDirection));
      updated = true;
    };
    if (this.keysDown[this.keys.zoom_in]) {
      this.cameraZoom -= this.cameraZoomSpeed * this.delta;

      if (this.cameraZoom < 0)
        this.cameraZoom = 0;
      if (this.cameraZoom > 100)
        this.cameraZoom = 100;

      if (this.cameraName == "persp") {
        this.camera.fov = this.minFov + (this.maxFov - this.minFov) *
          this.cameraZoom / 100;
      } else {
        var z = (this.cameraZoom > 0) ? this.cameraZoom : 1;
        this.camera.left = -this.rendererW / (100) * z;
        this.camera.right = this.rendererW / (100) * z;
        this.camera.top = this.rendererH / (100) * z;
        this.camera.bottom = -this.rendererH / (100) * z;
      };

      this.camera.updateProjectionMatrix();

      updated = true;
    };
    if (this.keysDown[this.keys.zoom_out]) {
      this.cameraZoom += this.cameraZoomSpeed * this.delta;

      if (this.cameraZoom < 0)
        this.cameraZoom = 0;
      if (this.cameraZoom > 100)
        this.cameraZoom = 100;

      if (this.cameraName == "persp") {
        this.camera.fov = this.minFov + (this.maxFov - this.minFov) *
          this.cameraZoom / 100;
      } else {
        var z = (this.cameraZoom > 0) ? this.cameraZoom : 1;
        this.camera.left = -this.rendererW / (100) * z;
        this.camera.right = this.rendererW / (100) * z;
        this.camera.top = this.rendererH / (100) * z;
        this.camera.bottom = -this.rendererH / (100) * z;
      };

      this.camera.updateProjectionMatrix();

      updated = true;
    };
    if (this.keysDown[this.keys.up]) {
      this.camera.translateOnAxis(this.upVector,
                                  this.cameraTranslation * this.delta);
      updated = true;
    };
    if (this.keysDown[this.keys.down]) {
      this.camera.translateOnAxis(this.upVector,
                                  -this.cameraTranslation * this.delta);
      updated = true;
    };
  },

  moveCameraFixed: function() {
    if (this.keysDown[this.keys.forward]) {
      this.cameraAngleVertical += this.cameraRotation * 180 /
        Math.PI * this.delta;
      if (this.cameraAngleVertical < -89)
        this.cameraAngleVertical = -89;
      if (this.cameraAngleVertical > 89)
        this.cameraAngleVertical = 89;
      this.updateCameraPosition();
      updated = true;
    };
    if (this.keysDown[this.keys.backward]) {
      this.cameraAngleVertical -= this.cameraRotation * 180 /
        Math.PI * this.delta;
      if (this.cameraAngleVertical < -89)
        this.cameraAngleVertical = -89;
      if (this.cameraAngleVertical > 89)
        this.cameraAngleVertical = 89;
      this.updateCameraPosition();
      updated = true;
    };
    if (this.keysDown[this.keys.straf_left]) {
      this.cameraAngleHorizontal += this.cameraRotation * 180 /
        Math.PI * this.delta;
      this.updateCameraPosition();
      updated = true;
    };
    if (this.keysDown[this.keys.straf_right]) {
      this.cameraAngleHorizontal -= this.cameraRotation * 180 /
        Math.PI * this.delta;
      this.updateCameraPosition();
      updated = true;
    };
    if (this.keysDown[this.keys.rot_up]) {
      this.cameraDistance -= this.cameraTranslation * this.delta;
      if (this.cameraDistance < 10)
        this.cameraDistance = 10;
      this.updateCameraPosition();
      updated = true;
    };
    if (this.keysDown[this.keys.rot_down]) {
      this.cameraDistance += this.cameraTranslation * this.delta;
      if (this.cameraDistance < 10)
        this.cameraDistance = 10;
      this.updateCameraPosition();
      updated = true;
    };
    if (this.keysDown[this.keys.zoom_in]) {
      this.cameraZoom -= this.cameraZoomSpeed * this.delta;

      if (this.cameraZoom < 0)
        this.cameraZoom = 0;
      if (this.cameraZoom > 100)
        this.cameraZoom = 100;

      if (this.cameraName == "persp") {
        this.camera.fov = this.minFov + (this.maxFov - this.minFov) *
          this.cameraZoom / 100;
      } else {
        var z = (this.cameraZoom > 0) ? this.cameraZoom : 1;
        this.camera.left = -this.rendererW / (100) * z;
        this.camera.right = this.rendererW / (100) * z;
        this.camera.top = this.rendererH / (100) * z;
        this.camera.bottom = -this.rendererH / (100) * z;
      };

      this.camera.updateProjectionMatrix();

      updated = true;
    };
    if (this.keysDown[this.keys.zoom_out]) {
      this.cameraZoom += this.cameraZoomSpeed * this.delta;

      if (this.cameraZoom < 0)
        this.cameraZoom = 0;
      if (this.cameraZoom > 100)
        this.cameraZoom = 100;

      if (this.cameraName == "persp") {
        this.camera.fov = this.minFov + (this.maxFov - this.minFov) *
          this.cameraZoom / 100;
      } else {
        var z = (this.cameraZoom > 0) ? this.cameraZoom : 1;
        this.camera.left = -this.rendererW / (100) * z;
        this.camera.right = this.rendererW / (100) * z;
        this.camera.top = this.rendererH / (100) * z;
        this.camera.bottom = -this.rendererH / (100) * z;
      };

      this.camera.updateProjectionMatrix();

      updated = true;
    };
  },

  update: function() {
    var now = new Date()
      .getTime();
    this.delta = (now - this.lastUpdate) / 1000;
    this.lastUpdate = now;
    this.keyEventHandler();
  },

  moveCameraTop: function() {
    this.cameraAngleHorizontal = 0;
    this.cameraAngleVertical = 90;

    this.updateCameraPosition();

    updated = true;
  },

  moveCameraBottom: function() {
    this.cameraAngleHorizontal = 0;
    this.cameraAngleVertical = -90;

    this.updateCameraPosition();

    updated = true;
  },

  moveCameraSide: function() {
    this.cameraAngleHorizontal = 0;
    this.cameraAngleVertical = 0;

    this.updateCameraPosition();

    updated = true;
  },

  resetCamera: function() {
    this.cameraDistance = 600;
    this.cameraAngleHorizontal = 50;
    this.cameraAngleVertical = 30;
    this.cameraZoom = 50;

    this.updateCameraPosition();

    updated = true;
  },

  handleKeyDown: function(e) {
    var key = e.keyCode;
    for (var k in this.keys) {
      if (key == this.keys[k]) {
        e.preventDefault();
        this.keysDown[e.keyCode] = true;
        break;
      };
    };
  },

  handleKeyUp: function(e) {
    var key = e.keyCode;
    for (var k in this.keys) {
      if (key == this.keys[k]) {
        e.preventDefault();
        this.keysDown[e.keyCode] = false;
        break;
      };
    };
  },

  changeCameraControls: function(t, v) {
    switch (v) {
      case 0:
        if (this.azerty) {
          this.azerty = false;
          this.keys.forward = this.keyCodes.w;
          this.keys.straf_left = this.keyCodes.a;
          this.keys.zoom_out = this.keyCodes.q;
          t.innerHTML = "Azerty";
        } else if (!this.azerty) {
          this.azerty = true;
          this.keys.forward = this.keyCodes.z;
          this.keys.straf_left = this.keyCodes.q;
          this.keys.zoom_out = this.keyCodes.a;
          t.innerHTML = "Qwertz";
        };
        break;
      case 1:
        if (this.freeFly) {
          this.freeFly = false;
          this.keyEventHandler = this.moveCameraFixed;
          this.setupFixedCamera();
          t.innerHTML = "Free";
        } else if (!this.freeFly) {
          this.freeFly = true;
          this.keyEventHandler = this.moveCameraFree;
          this.setupFreeCamera();
          t.innerHTML = "Fixed";
        };
        break;
    };

    updated = true;
  },

  setupFreeCamera: function() {
    this.cameraDirection = new THREE.Vector3(-this.camera.position.x,
                                             -this.camera.position.y,
                                             -this.camera.position.z)
      .normalize();
    var x = -this.camera.position.x
      , z = -this.camera.position.z;
    if (x == 0 && z == 0) {
      this.cameraLateralDirection = new THREE.Vector3(0, 0, 1)
        .normalize();
    } else {
      this.cameraLateralDirection = new THREE.Vector3(x, 0, z)
        .applyAxisAngle(new THREE.Vector3(0, -1, 0), Math.PI / 2)
        .normalize();
    };
  },

  setupFixedCamera: function() {
    this.camera.lookAt(this.origin);
    this.cameraDistance = this.camera.position.distanceTo(this.origin);
    this.cameraAngleHorizontal =
      Math.round(this.rightVector.angleTo(this.camera.position.clone()
      .setY(0)) * 180 / Math.PI);
    if (this.camera.position.z < 0)
      this.cameraAngleHorizontal = -this.cameraAngleHorizontal;
    this.cameraAngleVertical = 90 - Math.round(
      this.camera.position.angleTo(new THREE.Vector3(0, 1, 0)) * 180 / Math.PI);
  },

  updateCameraPosition: function() {
    this.camera.position.x = Math.floor(
      Math.cos(this.cameraAngleHorizontal * Math.PI / 180) *
      Math.cos(this.cameraAngleVertical * Math.PI / 180) * this.cameraDistance);
    this.camera.position.z = Math.floor(
      Math.sin(this.cameraAngleHorizontal * Math.PI / 180) *
      Math.cos(this.cameraAngleVertical * Math.PI / 180) * this.cameraDistance);
    this.camera.position.y = Math.floor(
      Math.sin(this.cameraAngleVertical * Math.PI / 180) * this.cameraDistance);

    this.cameraDirection = new THREE.Vector3(-this.camera.position.x,
                                             -this.camera.position.y,
                                             -this.camera.position.z)
      .normalize();

    var x = -this.camera.position.x
      , z = -this.camera.position.z;
    if (x == 0 && z == 0) {
      this.cameraLateralDirection = new THREE.Vector3(0, 0, 1)
        .normalize();
    } else {
      this.cameraLateralDirection = new THREE.Vector3(x, 0, z)
        .applyAxisAngle(new THREE.Vector3(0, -1, 0), Math.PI / 2)
        .normalize();
    }

    if (this.cameraName == "persp") {
      this.camera.fov = this.minFov + (this.maxFov - this.minFov) *
        this.cameraZoom / 100;
    } else {
      var z = (this.cameraZoom > 0) ? this.cameraZoom : 1;
      this.camera.left = -this.rendererW / (100) * z;
      this.camera.right = this.rendererW / (100) * z;
      this.camera.top = this.rendererH / (100) * z;
      this.camera.bottom = -this.rendererH / (100) * z;
    }

    this.camera.updateProjectionMatrix();

    this.camera.lookAt(this.origin);
  },

  updateLFP: function(index) {
    var data = this.data.lfpDataset[index];
    //TODO
    var i, j, dayaNow;
    var x = this.data.lfpXSize, y = this.data.lfpYSize;

    this.lfpCtx.fillStyle = "rgb(255,0,0)";
    this.lfpCtx.fillRect(0, 0, x, y);
    for (i = 0; i < y; i++) {
      for (j = 0; j < x; j++) {
        dataNow = this.data.lfpDataset[index][i * x + j];
        this.lfpCtx.fillStyle = this.colorLFP(dataNow);
        this.lfpCtx.fillRect(j, i, 1, 1);
      }
    }
    this.lfpTexture.needsUpdate = true;
  },


  colorLFP: function(v) {
    // diverging color map

    // rgb colors, currently using PRGn color map
    var cNeg = [64, 0, 75]; // dark purple
    var c0 = [255, 255, 255]; // white
    var cPos = [0, 68, 27]; // dark green

    // data mainly in [-lfpWidth,+lfpWidth] with lfpWidth=2*stdDev,
    // now scaled and bound to [-1, 1]
    var val = v / this.data.lfpWidth;
    if (val < -1) val = -1;
    if (val > 1) val = 1;

    if (val < 0) { // linear scaling between cNeg and c0
        return "rgb(" + Math.floor(c0[0]+(c0[0]-cNeg[0])*val) + "," +
                        Math.floor(c0[1]+(c0[1]-cNeg[1])*val) + "," +
                        Math.floor(c0[2]+(c0[2]-cNeg[2])*val) + ")";

    }
    else { // linear scaling between c0 and cPos
        return "rgb(" + Math.floor(c0[0]+(cPos[0]-c0[0])*val) + "," +
                        Math.floor(c0[1]+(cPos[1]-c0[1])*val) + "," +
                        Math.floor(c0[2]+(cPos[2]-c0[2])*val) + ")";
    }
  },


  updatePlane: function(index, k) {
    if (this.displayPop[k]) {

      if (this.data.dataType == "binned") {

        var currData = this.data.datasets[k][index];
        var sumData = this.data.sumDatasets[k][index];
        var rData;
        var logMax = Math.log(1 + this.data.maxSpikesAmount);
        var logData;

        for (var i = 0; i < this.xSize; i++) {
          for (var j = 0; j < this.ySize; j++) {
            logData = Math.log(1 +
              this.data.getScaledValue(currData[i * this.ySize + j])) / logMax;
            // third root of scaled spike count values
            rData = Math.pow(this.data.getScaledValue(currData[i * this.ySize + j]) /
              this.data.maxSpikesCutoff, 1./3.);

            if (rData > 0) {
              this.voxels[k][j + i * this.ySize].visible = true;
              if (this.useSize) {
                this.voxels[k][j + i * this.ySize].scale.set(rData,
                                                             rData,
                                                             rData);
              };
              if (this.usePosition) {
                if (this.logPos) {
                  if (this.groupLayers)
                    this.voxels[k][j + i * this.ySize].position.y =
                      logData * this.vDisplacement * this.boxSize *
                      (1 + (-k % 2) * 2);
                  else
                    this.voxels[k][j + i * this.ySize].position.y =
                      logData * this.vDisplacement * this.boxSize;
                } else {
                  if (this.groupLayers)
                    this.voxels[k][j + i * this.ySize].position.y =
                      rData * this.vDisplacement * this.boxSize *
                      (1 + (-k % 2) * 2);
                  else
                    this.voxels[k][j + i * this.ySize].position.y =
                      rData * this.vDisplacement * this.boxSize;
                };
              };
              if (this.useOpacity) {
                this.voxels[k][j + i * this.ySize].material.opacity = rData;
              };
            } else if (!this.hideInactiveBins) {
              this.voxels[k][j + i * this.ySize].visible = true;
              this.voxels[k][j + i * this.ySize].position.y = 0;
              this.voxels[k][j + i * this.ySize].material.opacity = 1;
            } else {
              this.voxels[k][j + i * this.ySize].visible = false;
            };
          };
        };

        //Update sumDisplays
        for (var i = 0; i < this.xSize + this.ySize; i++) {
          rData = this.data.getScaledValue(sumData[i]) / this.data.maxSumAmount;
          this.sumSquares[k][i].scale.set(1, 0.2 + this.sumScale * rData, 1);
          this.sumSquares[k][i].material.opacity = 1;
        };

      } else if (this.data.dataType == "neuron") {

        //Hide all neurons
        var indices = this.data.neuronLayers[k];
        for (i = 0; i < indices.length; i++) {
          if (this.voxels[indices[i]] !== undefined)
            this.voxels[indices[i]].visible = false;
        };

        var spikes;

        if (this.data.posReady[k]) {
          spikes = this.data.neuronSpikes[k][index];
          if (spikes !== undefined) {
            var limit = spikes.length * this.density / 100;
            for (var i = 0; i < limit; i++) {
              this.voxels[spikes[i]].visible = true;
            };
          };
        };
      };
    };
  },

  updateGLColor: function(c) {
    if (this.data.dataType == "binned") {

      var limit = this.voxels[c].length;
      for (var i = 0; i < limit; i++) {
        for (var j = 0; j < this.ySize; j++) {
          this.voxels[c][i].material.color =
            new THREE.Color(this.data.layerColors[c]);
        };
      };

      for (var i = 0; i < this.xSize + this.ySize; i++) {
        this.sumSquares[c][i].material.color =
          new THREE.Color(this.data.layerColors[c]);
      };



      this.timelines[c].material.color =
        new THREE.Color(this.data.layerColors[c]);
    } else if (this.data.dataType == "neuron") {
      var indices = this.data.neuronLayers[c];
      for (i = 0; i < indices.length; i++) {
        if (this.voxels[indices[i]] !== undefined)
          this.voxels[indices[i]].material.color =
            new THREE.Color(this.data.layerColors[c]);
      };
    };

    this.lMarkers[c].material.color = new THREE.Color(this.data.layerColors[c]);
  },

  updateBackgroundColor: function(color) {
    this.renderer.setClearColor("#" + color, 1);
    updated = true;
  },

  updateBoxColor: function(color) {
    this.timelineBoundingBox.material.color = new THREE.Color("#" + color);
    updated = true;
  },

  updateSumBackgroundColor: function(color) {
    this.plane.material.color = new THREE.Color("#" + color);
    this.plane2.material.color = new THREE.Color("#" + color);
    updated = true;
  },

  updateTimelineIndex: function(index) {

    if (index < this.zTimeSize / 2) {
      this.timeOffset = 0;
    } else if (index >= this.data.timestamps - this.zTimeSize / 2) {
      this.timeOffset = this.data.timestamps - this.zTimeSize;
    } else {
      this.timeOffset = Math.floor(index - this.zTimeSize / 2);
    };

    for (var i = 0; i < this.data.nLayers; i++) {
      this.timelines[i].setTimeOffset(this.timeOffset);
    };

    this.timelineIndex.position.z =
      Math.floor(index - this.timeOffset - this.zTimeSize / 2) /
      this.zTimeSize * 2 * 100 *
      (this.zTimeSize - 2) / (this.data.xNeurons - 2);
      /* (index - this.timeOffset - this.zTimeSize/2) goes from -20 to 20 and we
         scaled the geometry by 100, while the vertices
         have indices from -1 to 1, so 100 / 20 = 5 */
  },

  updateLayerPositions: function() {
    this.lNum = 0;
    //var p = 0;
    for (var k = 0; k < this.data.nLayers; k++) {
      if (this.displayPop[k]) {
        this.lPos[k] = this.lNum;
        this.lNum++;
      };
    };
    for (var k = 0; k < data.nLayers; k++) {
      if (this.displayPop[k]) {
        if (this.groupLayers) {
          this.yLayerPos[k] =
            Math.floor((this.lNum - 1) / 2) *
              this.boxSize * this.layerDisplacement / 2 +
              Math.floor(this.lNum / 2) *
              this.boxSize * this.groupDisplacement / 2 -
              this.boxSize * this.groupDisplacement *
              Math.ceil(this.lPos[k] / 2) -
              this.boxSize * this.layerDisplacement *
              Math.floor(this.lPos[k] / 2);
        } else {
          this.yLayerPos[k] = (this.lNum - 1) * this.boxSize *
            this.layerDisplacement / 2 - this.lPos[k] * this.boxSize *
            this.layerDisplacement;
        };
        this.oVoxels[k].position.y = this.yLayerPos[k];
        if (this.lfpBottom)
          this.lfpMesh.position.y = this.yLayerPos[k] - this.lfpSpacing;
      };
    };
    if (!this.lfpBottom) {
      this.lfpMesh.position.y = this.yLayerPos[this.lfpPosition[0]] +
        (this.yLayerPos[this.lfpPosition[1]] -
          this.yLayerPos[this.lfpPosition[0]]) / 2;
    };

    if (this.data.dataType == "binned")
      this.updateSumBackground();
  },

  togglePop: function(b, v) {
    if (b != this.displayPop[v]) {
      this.displayPop[v] = b;

      if (b)
        this.layersScene.add(this.oVoxels[v]);
      else
        this.layersScene.remove(this.oVoxels[v]);

      if (this.displayPop[this.lfpPosition[0]] &&
          this.displayPop[this.lfpPosition[1]]) {
        this.lfpBottom = false;
      } else {
        this.lfpBottom = true;
      };

      this.updateLayerPositions();
      this.renderScene();
    };
  },

  toggleTimelinePop: function(b, v) {
    this.timelines[v].visible = b;
    updated = true;
  },

  toggleSumDisplay: function(v) {
    if (v) {
      for (var i = 0; i < this.data.nLayers; i++) {
        this.oVoxels[i].remove(this.oSumSquares[i]);
      };
    } else {
      for (var i = 0; i < this.data.nLayers; i++) {
        this.oVoxels[i].add(this.oSumSquares[i]);
      };
    };
  },

  toggleSumBackground: function(v) {
    if (v) {
      this.layersScene.add(this.plane);
      this.layersScene.add(this.plane2);
    } else {
      this.layersScene.remove(this.plane);
      this.layersScene.remove(this.plane2);
    };
    this.renderScene();
  },

  setBackgroundScale: function(v) {
    this.backgroundScale = parseFloat(v);
    this.updateSumBackground();
    this.renderScene();
  },

  setNeuronSize: function(v) {
    this.boxSize = parseInt(v);

    //Recreate layerMarkers
    var markerG = new THREE.Geometry();
    markerG.vertices.push(
      new THREE.Vector3(-this.xSize / 2 - this.boxSize / 2,
                        0,
                        -this.ySize / 2 - this.boxSize / 2),
      new THREE.Vector3(-this.xSize / 2 - this.boxSize / 2,
                          0,
                          this.ySize / 2 + this.boxSize / 2),
      new THREE.Vector3(this.xSize / 2 + this.boxSize / 2,
                          0,
                          this.ySize / 2 + this.boxSize / 2),
      new THREE.Vector3(this.xSize / 2 + this.boxSize / 2,
                          0,
                          -this.ySize / 2 - this.boxSize / 2),
      new THREE.Vector3(-this.xSize / 2 - this.boxSize / 2,
                          0,
                          -this.ySize / 2 - this.boxSize / 2));


    for (var i = 0; i < this.data.nLayers; i++) {
      this.lMarkers[i] = new THREE.Line(markerG, new THREE.LineBasicMaterial({ color: this.data.layerColors[i] }));
      this.createNeuronGeometries(i);
    };
    this.updateLayerPositions();
    updated = true;

  },

  updateSumBackground: function() {
    var lowerLayer = -1
      , upperLayer = -1;
    for (var k = 0; k < data.nLayers; k++) {
      if (this.displayPop[k]) {
        if (lowerLayer == -1)
          lowerLayer = k;
        uppperLayer = k;
      };
    };
    var size = this.yLayerPos[lowerLayer] - this.yLayerPos[uppperLayer];
    if (size == 0)
      size = this.boxSize * 2;
    this.plane.scale.set(1,
                         this.backgroundScale * (size) / this.basePlaneHeight,
                         1);
    this.plane2.scale.set(1,
                          this.backgroundScale * (size) / this.basePlaneHeight,
                          1);
  }
};
