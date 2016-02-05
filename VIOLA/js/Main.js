//Setup file reader for JSON config file
(function(){
    var reader = new FileReader();
	var fileInput = document.getElementById("importConfigButton");
	fileInput.onchange = function (e) {
		e.preventDefault();
        
        var fName = fileInput.files[0].name.split(".");
		reader.onload = function() {
			if(fName[1] == "json"){
				importConfig(reader.result);
			}else{
				alert("Wrong file format !");
			}
		};
        
		reader.readAsText(fileInput.files[0]);

	};
	
})();

//Set dataType from the html value
updateDataType();

//Timers
var tStart,tEnd;

//App classes
var data,r3D,r2D,graph,corr;

//Booleans to know if a display is worth refreshing
var draw3D = false;
var draw2D = true;
var drawGraph = true;
var drawCorr = false;
//Collapse files panel
var collapsed = false;

var index = 0;
var animate = false;
var isBackwards = false;
var animateInterval;
var timeStep = 100;

var nr = new FileReader();
var r = [];
var holder;

var barW = 800;
var barH = 10;
var bar = document.getElementById("barCanvas")
var barC = bar.getContext("2d");
var timerW = 100;
var timerC = document.getElementById("timerCanvas").getContext("2d");
var dragging = false;

var updated = true;

function resetAll(){
	//Stop simulation
	if(animate){
		animateF();
	}
	//Reset index
	index = 0;
	//Reset colors
	data.layerColors = document.getElementById('popColors').value.split(",");
	for(var i=0; i<data.nLayers; i++){
		r3D.updateGLColor(i);
	}
	//Frame interval
	updateSpeed(100,true);
    
	//Redraw everything
	r3D.resetCamera();
	updated=true;
}

function setup(){
	var num = document.getElementById('popNumber').value;
	var popNames = document.getElementById('popNames').value.split(",");
	var popColors = document.getElementById('popColors').value.split(",");
        var spikesFiles = document.getElementById('spikesFiles').value.split(",");
	var timestamps = document.getElementById('timestampsNumber').value;
	var resolution = document.getElementById('resNumber').value;
	var xSize = document.getElementById('xSize').value;
	var ySize = document.getElementById('ySize').value;
        var zTimeSize = document.getElementById('zTimeSize').value;
        var dataType = document.getElementById('dataTypeSelect').value;
    if(dataType == "binned"){
	var xNumber = document.getElementById('xNumber').value;
	var yNumber = document.getElementById('yNumber').value;
        var lfpx = document.getElementById('lfpx').value;
        var lfpy = document.getElementById('lfpy').value;
    }else if(dataType == "neuron"){
        var posFiles = document.getElementById('posFiles').value.split(",");
    }
    
    //Test if config is valid
    if(num <= 0){
        alert("Number of populations must be superior to 0");
        return;
    }
    if(popNames.length != num){
        alert("Wrong number of population names");
        return;
    }
    if(popColors.length != num){
        alert("Wrong number of colors");
        return;
    }
    if(spikesFiles.length != num && spikesFiles != ""){
        alert("Wrong number of spikes files");
        return;
    }
    if(dataType == "neuron" && posFiles.length != num && posFiles != ""){
        alert("Wrong number of positions files");
        return;
    }

	document.getElementById('config-panel').className="hiddenPanel";
	data = new Visu.Data({layers:num,names:popNames, colors:popColors, timestamps:timestamps, resolution:resolution, x:xSize, y:ySize, xn:xNumber, yn:yNumber, lfpx:lfpx, lfpy:lfpy, ztimesize:zTimeSize, dataType:dataType, positions:posFiles, spikes:spikesFiles});
	init();

}


//--------------------------------------------------------------------------------------------------------
//All those functions serve to generate the parameters inputs according to the settings the user specified
//--------------------------------------------------------------------------------------------------------
function generateUploadPanel(){

	var ih = '<table id="filesTable"><caption style="color:black;text-align:center">Files to upload</caption><tr><th>Animation Data</th></tr>';
    
	for(var i=0; i<data.nLayers; i++){
		ih+='<tr><td id="hn'+data.layerNames[i]+'"><span>'+data.layerNames[i]
		+'</span><br/><span class="fileName" id="sn'+data.layerNames[i]+'"></span></td>'
		+'</tr>';

        if(data.dataType == "neuron"){
            ih += '<tr><td id="pos'+data.layerNames[i]+'"><span>'+data.layerNames[i]+' Positions</span><br/>'
            +'<span class="fileName" id="posn'+data.layerNames[i]+'"></span></td></tr>';
        }}
        if(data.dataType == "binned"){
            ih+='<tr><td id="lfp"><span>LFP</span><br/><span class="fileName" id="lfpn"></span></td></tr>';
        }
	ih+='</table>';
	document.getElementById("fileNames").innerHTML=ih;
}

function generateTimelineParams(){
    var ih = '';
    ih += '<ul>';
	for(var i=0; i<data.nLayers; i++){
		ih+='<li><input id="timelinePopCheckBox'+i+'" disabled type="checkbox" onchange="r3D.toggleTimelinePop(this.checked,this.value)" value="'+i+'">'+data.layerNames[i]+'</li>';
	}
    ih += '</ul>';
    ih += '<span>Isolation (spikes per s) : </span><input type="range" min="1" max="1000" value="100" step="1" onmousedown="r3D.disableDrag(true)" onmouseup="r3D.disableDrag(false)" oninput="r3D.setIsolation(this.value);document.getElementById(\'isospan\').innerHTML=this.value" style="width:200px;vertical-align:middle;display:inline;margin: 0 10px 0"><span id="isospan">100</span>';
    ih +='<br/>';
    ih += '<span>Time window (timestamps) : </span><input id="timeWindowRange" type="range" min="1" max="1000" value="'+data.zTimeSize+'" step="1" onmousedown="r3D.disableDrag(true)" onmouseup="r3D.disableDrag(false)" oninput="r3D.setTimeWindow(this.value,0,index)" style="width:200px;vertical-align:middle;display:inline;margin: 0 10px 0"><input type="number"  id="timeWindowValue" onchange="r3D.setTimeWindow(this.value,1,index)" value="'+data.zTimeSize+'" style="width:45px;vertical-align:middle;display:inline">';
    ih+= '<br/><span>Delays (timestamps) :</span><span style="position:absolute;left:220px">Opacity :</span>';
    for(var i=0; i<data.nLayers; i++){
        ih += '<br/>';
        ih += '<span>'+data.layerNames[i]+'</span><input id="delayRange'+i+'" type="range" min="-5" max="5" value="0" step="1" onmousedown="r3D.disableDrag(true)" onmouseup="r3D.disableDrag(false)" oninput="r3D.setDelay(this.value,'+i+')" style="width:100px;vertical-align:middle;display:inline;margin: 0 10px 0;position:absolute;left:50px"><span style="position:absolute;left:180px" id="delayspan'+i+'">0</span>';
        ih +='<input id="opacityRange'+i+'" type="range" min="0" max="10" value="10" step="1" onmousedown="r3D.disableDrag(true)" onmouseup="r3D.disableDrag(false)" oninput="r3D.setOpacity(this.value,'+i+')" style="width:100px;vertical-align:middle;display:inline;margin: 0 10px 0;position:absolute;left:210px">'
    }
    
    document.getElementById("timelineParams").innerHTML=ih;
}

function generate3Dparams(){

	var ih = '<button onclick="hideAll()">Hide all</button><button onclick="showAll()">Show all</button>';
    ih+='<ul>';
    ih+= '<li><input id="lfpCheckbox" type="checkbox" checked onchange="r3D.toggleLFPs(this.checked)">LFPs</li>';
	for(var i=0; i<data.nLayers; i++){
		ih+='<li><input id="'+'popCheckBox'+i+'" type="checkbox" checked onchange="r3D.togglePop(this.checked,this.value)" value="'+i+'">'+data.layerNames[i]+'</li>';
	}
	ih+='<li><input type="checkbox" onchange="r3D.toggleAxis(this.checked)">Axes</li>'
	+'<li><input type="checkbox" checked onchange="r3D.toggleTags(this.checked)">Name tags</li>'
	+'<li><input type="checkbox" checked onchange="r3D.toggleMarkers(this.checked)">Layer markers</li>'
    +'</ul>';
    
    if(data.dataType == "binned"){
	    ih+='<ul><span>Use log scales : </span>'
	    +'<li><input type="checkbox" onchange="r3D.toggleLogPos(this.checked)">Position</li>'
        +'</ul>'
	    +'<ul><span>Sum display settings : </span>'
	    +'<li><input type="checkbox" onchange="r3D.toggleSumDisplay(this.checked)">Hide</li>'
        +'<li><input id="sumScale" type="number" value="10" style="width:40px" onchange="r3D.setSumScale(this.value)">Scale</li>'
	    +'<li><input type="checkbox" checked onchange="r3D.toggleSumBackground(this.checked)">Background</li>'
        +'<li><input id="sumScale" type="number" value="1" style="width:40px" onchange="r3D.setBackgroundScale(this.value)">Background scale</li>'
        +'</ul>'
		+'<ul><span>Display part of the layers : </span>'
		+'<li>X from<input id="xMin" type="number" value="0" style="width:40px" onchange="r3D.setMinX(this.value)">to</li>'
		+'<li><input id="xMax" type="number" value="'+data.xNeurons+'" style="width:40px" onchange="r3D.setMaxX(this.value)"></li>'
		+'<li>Y from<input id="yMin" type="number" value="0" style="width:40px" onchange="r3D.setMinY(this.value)">to</li>'
		+'<li><input id="yMax" type="number" value="'+data.yNeurons+'" style="width:40px" onchange="r3D.setMaxY(this.value)"></li>'
		+'<li><button onclick="r3D.adjustDisplayedNeurons()">Update view</li></ul>';
    }else if(data.dataType == "neuron"){
        ih+='<ul><li><span>Display density </span><input onmousedown="r3D.disableDrag(true)" onmouseup="r3D.disableDrag(false)" type="range" min="0" max="100" value="100" step="1" oninput="r3D.setDensity(this.value)" style="width:200px;vertical-align:middle;display:inline;margin: 0 10px 0"></li>';
        ih+='<li><input id="neuronDisplaySize" type="number" value="4" style="width:40px" onchange="r3D.setNeuronSize(this.value)"></li></ul>'
    }
	
	document.getElementById("layersParams").innerHTML=ih;
	
	ih = '<ul><span>Camera : </span><li></li>'
    +'<li><button id="topButton" onclick="r3D.moveCameraTop()"><span>Top</span></button></li>'
	+'<li><button id="bottomButton" onclick="r3D.moveCameraBottom()"><span>Bottom</span></button></li>'
	+'<li><button id="sideButton" onclick="r3D.moveCameraSide()"><span>Side</span></button></li>'
	+'<li><button id="resetCameraButton" onclick="r3D.resetCamera()"><span>Reset</span></button></li>'
	+'</ul>';
	document.getElementById("cameraControls").innerHTML=ih;
}

function generateGraphParams(){
	var ih = '<ul>';
	for(var i=0; i<data.nLayers; i++){
		ih+='<li><input id="graphPopCheck'+i+'" type="checkbox" checked onchange="graph.togglePop(this.checked,this.value)" value="'+i+'">'+data.layerNames[i]+'</li>';
	}
	ih+='</ul>';
	
	document.getElementById("graphSelectPop").innerHTML=ih;							
}


//-------------------------------------------------------------------------
//Those functions deal with reading the different files used in the program
//-------------------------------------------------------------------------
function importConfig(text){
    var config = JSON.parse(text);
    
	document.getElementById('popNumber').value = config.popNum;
	document.getElementById('popNames').value = config.popNames;
    document.getElementById('spikesFiles').value = config.spikesFiles;
	document.getElementById('timestampsNumber').value = config.timestamps;
	document.getElementById('resNumber').value = config.resolution;
	document.getElementById('xSize').value = config.xSize;
	document.getElementById('ySize').value = config.ySize;
    document.getElementById("dataTypeSelect").value = config.dataType;
    updateDataType();
    if(config.dataType == "binned"){
        document.getElementById('xNumber').value = config.xBins;
        document.getElementById('yNumber').value = config.yBins;
        document.getElementById('lfpx').value = config.xLFP;
        document.getElementById('lfpy').value = config.yLFP;
    }else if(config.dataType == "neuron"){
        document.getElementById('posFiles').value = config.posFiles;
    }
    document.getElementById('zTimeSize').value = config.timelineLenght;
	document.getElementById('popColors').value = config.popColors;
}

function readFile(f,reader,n,file){
    tStart = new Date();
	f.call(data,reader.result,n);
	data.mReady[n]=true;
    data.computeRMS();
	if(data.dataType == "binned"){
		graph.readData(n);
	}else if(data.dataType == "neuron"){
		graph.readNeuronData(n);
	}
	graph.computeData();
	tEnd = new Date();
	console.log("data processed in "+(tEnd.getTime()-tStart.getTime())+" ms");
	document.getElementById('hn'+data.layerNames[n]).className='done';
	document.getElementById('sn'+data.layerNames[n]).innerHTML="'"+file.name+"'";
	document.getElementById('hn'+data.layerNames[n]).ondragleave = function () { this.className = 'done'; return false; };
    document.getElementById('timelinePopCheckBox'+n).disabled = false;
	graph.displayData();
	graph.drawAxes();
	if(data.dataType == "binned"){
		r2D.drawMiniLegend();
	}
}

function readCompact2File(reader,n,file){
	readFile(data.dataCompact2Gen,reader,n,file);
}

function readLFPFile(reader,file){
    tStart = new Date();
	data.lfpDataGen(reader.result);
	data.lfpReady=true;
	tEnd = new Date();
    document.getElementById('lfp').className='done';
    document.getElementById('lfp').ondragleave = function () { this.className = 'done'; return false; };
    document.getElementById('lfpn').innerHTML="'"+file.name+"'";
	console.log("data processed in "+(tEnd.getTime()-tStart.getTime())+" ms");
}

function readGDFFile(reader,n,file){
	readFile(data.importSpikes,reader,n,file);
}

function readRawPosFile(reader,n,file){
    tStart = new Date();
	data.importPos(reader.result,n);
	data.posReady[n]=true;
    r3D.createNeuronGeometries(n);
	tEnd = new Date();
    document.getElementById('pos'+data.layerNames[n]).className='done';
    document.getElementById('pos'+data.layerNames[n]).ondragleave = function () { this.className = 'done'; return false; };
    document.getElementById('posn'+data.layerNames[n]).innerHTML="'"+file.name+"'";
	console.log("data processed in "+(tEnd.getTime()-tStart.getTime())+" ms");
}

function readerOnLoadCallback(reader,files,i,h){
	var fName = files[i].name.split(".");
	console.log("Reading file of type "+fName[1]+", "+(files.length-i)+"/"+files.length);
	if(fName[1] == "dat" && data.dataType == "binned"){
        var s = fName[0].split("_");
        var popName = s[s.length-1];
		var n = data.layerNames.indexOf(popName);
		if(n == -1){
            n = data.spikesFiles.indexOf(files[i].name);
            if(n == -1){
			    alert("No population named : "+popName);
            }else{
                readCompact2File(reader,n,files[i]);
            }
		}else{
			readCompact2File(reader,n,files[i]);
		}
	}else if(fName[1] == "gdf" && data.dataType == "neuron"){
		var n = data.spikesFiles.indexOf(files[i].name);
		if(n == -1){
			alert("No population corresponding to the file : "+files[i].name);
		}else{
			readGDFFile(reader,n,files[i]);
		}
	}else if(fName[1] == "dat" && data.dataType == "neuron"){
		var n = data.posFiles.indexOf(files[i].name);
		if(n == -1){
			alert("No population corresponding to the file : "+files[i].name);
		}else{
			readRawPosFile(reader,n,files[i]);
		}
	}else if(fName[1] == "lfp" && data.dataType == "binned"){
        readLFPFile(reader,files[i]);
    }else{
		alert("Wrong file format !");
	}

	i--;
	if(i>=0)
		return i;
	else{
		h.className = '';
	    return -1
	}
}

function init(){

    //Add key listeners
	window.addEventListener("keypress",handleKeyPress);
    window.addEventListener("keydown",handleKeyDown);
    window.addEventListener("keyup",handleKeyUp);
    
	document.getElementById('main-container').className="";

	generateUploadPanel();
	generate3Dparams();
    generateTimelineParams();
	generateGraphParams();
	
	//Disable correlation panel menu for neuron data
	if(data.dataType == "neuron"){
		document.getElementById("corrmenudiv").parentNode.style.display="none";
		document.getElementById("sceneMenuItem").parentNode.style.display="none";
	}


    //---------------------------------------------
    //Create callbacks for importing the data files
    //---------------------------------------------
	(function(reader){

	    //Multidrop callback setup
        //
		//Hover modification
		holder = document.getElementById('multiDragInput');
		holder.ondragover = function () { this.className = 'hover'; return false; };
		holder.ondragleave = function () { this.className = ''; return false; };
		
		//Listeners for file loading
		holder.ondrop = function (e) {
			e.preventDefault();
			
			console.log(e.dataTransfer.files);
			
			(function(files,h){
				console.log("number of files "+files.length);
                var i = files.length-1;
				reader.onload = function() {   
                    i = readerOnLoadCallback(reader,files,i,h);
                    if(i >= 0)
                        reader.readAsText(files[i]);
				};
				reader.readAsText(files[files.length-1]);
				return false;					
			})(e.dataTransfer.files,this);
		};
        
        //Import button callback setup
		var fileInput = document.getElementById("uploadButton");
		fileInput.onchange = function (e) {
			e.preventDefault();
			
			console.log(fileInput.files);
			
			document.getElementById('multiDragInput').className = 'hover';

			(function(files,h){
				var i=files.length-1;
				console.log("number of files "+files.length);
                if(files.length>0){
					reader.onload = function() {   
                        i = readerOnLoadCallback(reader,files,i,h);
                        if(i >= 0)
                            reader.readAsText(files[i]);
					};
					reader.readAsText(files[i]);
                }else{
                    h.className = '';
                }
				return false;					
			})(fileInput.files,document.getElementById('multiDragInput'));
		};
		
	})(nr);
    
    //LFP file table row
    if(data.dataType == "binned"){
	(function(reader){
		//Hover modification
		holder = document.getElementById('lfp');
		holder.ondragover = function () { this.className = 'hover'; return false; };
		holder.ondragleave = function () { this.className = ''; return false; };

		//Listeners for file loading
		holder.ondrop = function (e) {
			e.preventDefault();

			if(e.dataTransfer.files.length == 0){
				this.className = '';
				return false;
			}
			if(e.dataTransfer.files[0].name.split(".")[1] != "lfp"){
			alert("Wrong file format !");
				this.className = '';
				return false;
			}
		
			var file = e.dataTransfer.files[0];
			reader.onload = function() {
				readLFPFile(reader,file);
			};
			reader.readAsText(file);
			this.ondragleave = function () { this.className = 'done'; return false; };
			return false;
		};
	})(nr);
    }

    //Single file drop
	for(var j=0; j<data.nLayers; j++){
        if(data.dataType == "binned"){
			(function(reader,n){
        		//Hover modification
        		holder = document.getElementById('hn'+data.layerNames[n]);
        		holder.ondragover = function () { this.className = 'hover'; return false; };
        		holder.ondragleave = function () { this.className = ''; return false; };
	
        		//Listeners for file loading
        		holder.ondrop = function (e) {
        			e.preventDefault();

        			if(e.dataTransfer.files.length == 0){
        				this.className = '';
        				return false;
        			}
                    
                    var files = e.dataTransfer.files;
        			reader.onload = function() {
    					readerOnLoadCallback(reader,files,0,this);
        			};
        			reader.readAsText(files[0]);
        			this.ondragleave = function () { this.className = 'done'; return false; };
        			return false;
        		};
			})(nr,j);
        }
        
        if(data.dataType == "neuron"){
			(function(reader,n){
        		//Hover modification
        		holder = document.getElementById('hn'+data.layerNames[n]);
        		holder.ondragover = function () { this.className = 'hover'; return false; };
        		holder.ondragleave = function () { this.className = ''; return false; };
	
        		//Listeners for file loading
        		holder.ondrop = function (e) {
        			e.preventDefault();

        			if(e.dataTransfer.files.length == 0){
        				this.className = '';
        				return false;
        			}
			
        			var file = e.dataTransfer.files[0];
                    var fName = file.name.split(".");
        			reader.onload = function() {
    					if(fName[1] == "gdf"){
    						readGDFFile(reader,n,file);
						}else{
    						alert("Wrong file format !");
    					}
        			};
        			reader.readAsText(file);
        			this.ondragleave = function () { this.className = 'done'; return false; };
        			return false;
        		};
			})(nr,j);
            //Positions row
    		(function(reader,n){
        		//Hover modification
        		holder = document.getElementById('pos'+data.layerNames[n]);
        		holder.ondragover = function () { this.className = 'hover'; return false; };
        		holder.ondragleave = function () { this.className = ''; return false; };

        		//Listeners for file loading
        		holder.ondrop = function (e) {
        			e.preventDefault();

        			if(e.dataTransfer.files.length == 0){
        				this.className = '';
        				return false;
        			}
		
        			var file = e.dataTransfer.files[0];
                    var fName = file.name.split(".");
        			reader.onload = function() {
    					if(fName[1] == "dat"){
        				    readRawPosFile(reader,n,file);
						}else{
    						alert("Wrong file format !");
    					}
        			};
        			reader.readAsText(file);
        			this.ondragleave = function () { this.className = 'done'; return false; };
        			return false;
        		};
    		})(nr,j);
        }
	}
		
	//Color Selector
	var colorSelector = document.getElementById("popColorSelect")
	var colorInner = "";
	for(var i = 0 ; i < data.nLayers ; i++){
		colorInner+="\<option value=\""+i+"\">"+data.layerNames[i]+"\</option>";
	}
	colorSelector.innerHTML=colorInner;
	updateSelector(0);

	r3D = new Visu.Renderer3D(document.getElementById('webGLPanel'),data);
	
	r2D = new Visu.Renderer2D(document.getElementById('miniCanvasPanel'),data,"r2D");

	graph = new Visu.Graph(document.getElementById('graphCanvasPanel'),data);
	graph.drawLegend();
	
	corr = new Visu.Correlation(document.getElementById('correlationCanvasPanel'),data);
	//Add options for select layer
	var ih = "";
	for(var i = 0 ; i < data.nLayers ; i++){
		ih+="\<option value=\""+i+"\">"+data.layerNames[i]+"\</option>";
	}
	document.getElementById("popCorr1").innerHTML=ih;
	document.getElementById("popCorr2").innerHTML=ih;

	r2D.drawMiniLegends();
    
    corr.draw();
	
    render();
    
    setupMenuItems();

}

function hideAll(){
	for(var i=0; i<data.nLayers; i++){
        document.getElementById('popCheckBox'+i).checked=false;
		r3D.togglePop(false,i);
	}
    document.getElementById('lfpCheckbox').checked = false;
    r3D.toggleLFPs(false);
}

function showAll(){
	for(var i=0; i<data.nLayers; i++){
        document.getElementById('popCheckBox'+i).checked=true;
		r3D.togglePop(true,i);
	}
    document.getElementById('lfpCheckbox').checked = true;
    r3D.toggleLFPs(true);
}

		
//Collapse side panel
function collapsePanel(){
	if(collapsed){
		document.getElementById("files-container").style.display="";
		collapsed=false;
	}else{
		document.getElementById("files-container").style.display="none";
		collapsed=true;
	}
}

function togglePanel(p){
	var d;
	if(p.id == "2dmenudiv"){
        draw2D = !draw2D;
        d = draw2D;
        p = document.getElementById("miniCanvasPanel");
	}else if(p.id == "3dmenudiv"){
        draw3D = !draw3D;
        d = draw3D;
        p = document.getElementById("webGLPanel");
	}else if(p.id == "graphmenudiv"){
        drawGraph = !drawGraph;
        d = drawGraph;
        p = document.getElementById("graphCanvasPanel");
	}else if(p.id == "corrmenudiv"){
		drawCorr = !drawCorr;
		d = drawCorr;
        p = document.getElementById("correlationCanvasPanel");
	}
	if(d){
		p.className = "displayPanel";
	}else{
		p.className = "hiddenPanel";
	}
    updated = true;
}
	
function updateSelector(i){
	document.querySelector("#colorPicker").color.fromString(data.layerColors[i].substring(1));
}
	
function updateColor(c){
	data.layerColors[document.getElementById("popColorSelect").value]="#"+c;
	r3D.updateGLColor(document.getElementById("popColorSelect").value);
    graph.drawLegend();
    updated = true;
}

function updateSpeed(s,c){
	if(c){
		timeStep = s;
		if(animate){
			clearInterval(animateInterval);
			if(isBackwards)
				animateInterval = setInterval(prevIndex,timeStep);
			else
				animateInterval = setInterval(nextIndex,timeStep);
		}
	}
	document.getElementById("speedValue").value = s;
	document.getElementById("speedInput").value = s;
}	
	
//---------------
//Draw main frame
//---------------
function draw(){

	data.setTime(index*data.resolution);

	r3D.setTime(data.currTime);
	
	graph.setOffset(index);
    if(drawGraph){
		graph.displayData(index);
	}
	drawIndex();
			
	if(draw2D){
        r2D.draw(index);
	}
    if(draw3D){
	    r3D.renderScene();
	}
}

function render(){
    requestAnimationFrame(render);
    update(); //Used to get the input keys
    if(updated){
        draw();
    }
    updated=false;
}

function updateData(){
    if(draw3D){
        if(r3D.visu == "layers"){
			for(var c=0; c < data.nLayers; c++){
				if(data.mReady[c]){
					r3D.updatePlane(index,c);
				}
			}
            if(data.lfpReady){
                r3D.updateLFP(index);
            }
        }else{
            r3D.updateTimelineIndex(index);
			for(var c=0; c < data.nLayers; c++){
				if(data.mReady[c]){
					r3D.timelines[c].generateData(data.datasets[c]);
				}
			}
            if(data.lfpReady){
                r3D.updateLFP(index);
            }
            r3D.updateTimelineIndex(index);
        }
    }
}

//Tell Visu3D it can update key entries
function update(){
    r3D.update();
}

//Go to next index
function nextIndex(){
	index = (index+1)%data.timestamps;
    updateData();
    updated=true;
	//draw();
}

//Go to previous index
function prevIndex(){
	index = (index+data.timestamps-1)%data.timestamps;
    updateData();
    updated=true;
	//draw();
}

//Change "direction" of the animation
function backwards(){
	if(animate){
		document.getElementById("backwardButton").disabled=true;
		document.getElementById("forwardButton").disabled=false;
		clearInterval(animateInterval);
		animateInterval = setInterval(prevIndex,timeStep);
		isBackwards=true;
	}else{
		prevIndex();
	}
}

//Change "direction" of the animation
function forward(){
	if(animate){
		document.getElementById("forwardButton").disabled=true;
		document.getElementById("backwardButton").disabled=false;
		clearInterval(animateInterval);
		animateInterval = setInterval(nextIndex,timeStep);
		isBackwards=false;
		
	}else{
		nextIndex();
	}
}

//Toggle between play and pause
function animateF(){
	if(animate){
		clearInterval(animateInterval);
		animate=false;
		document.getElementById("playpause").innerHTML = "Play";
		document.getElementById("backward").innerHTML = "&lt;";
		document.getElementById("forward").innerHTML = "&gt;";
		document.getElementById("backwardButton").disabled=false;
		document.getElementById("forwardButton").disabled=false;
		
	}else{
		nextIndex();
		animateInterval = setInterval(nextIndex,timeStep);
		animate=true;
		isBackwards=false;
		document.getElementById("forwardButton").disabled=true;
		document.getElementById("playpause").innerHTML = "Pause";
		document.getElementById("backward").innerHTML = "&lt;&lt;";
		document.getElementById("forward").innerHTML = "&gt;&gt;";
	}
	
}

//--------------------------
//        Controls
//--------------------------

bar.addEventListener("mousedown",barDown);

//Global event handling
function handleKeyPress(e){
	if(e.keyCode == 32){//Press space to play/pause the animation
		e.preventDefault();
		animateF();
	}
}

function handleKeyDown(e){
    if(e.shiftKey){
        if(e.keyCode == r3D.keyCodes.s){
            e.preventDefault();
            if(data.dataType == "binned")
                r3D.switchScene();
        }else if(e.keyCode == r3D.keyCodes.c){
            e.preventDefault();
            r3D.changeCameraControls(document.getElementById('cameraStyleButton'),1);
        }else if(e.keyCode == r3D.keyCodes.l){
            e.preventDefault();
            r3D.changeCameraControls(document.getElementById('layoutButton'),0);
        }else if(e.keyCode == r3D.keyCodes.m){
            e.preventDefault();
            r3D.switchCameraType();
        }
    }else if(!e.metaKey){
	    r3D.handleKeyDown(e);
    }
}

function handleKeyUp(e){
	r3D.handleKeyUp(e);
}

//Code to enable the user to drag the index in the bottom timeline
function barDown(e){
	window.addEventListener("mousemove",barMove);
    window.addEventListener("mouseup",barUp);
	e.preventDefault();
	index = Math.floor((e.clientX-bar.getBoundingClientRect().left)/barW*data.timestamps);
	if(index < 0)
		index = 0;
	if(index >= data.timestamps)
		index = data.timestamps-1;
    updateData();
	updated=true;
}

function barUp(e){
    window.removeEventListener("mousemove",barMove);
    window.removeEventListener("mouseup",barUp);
	updated=true;
}

function barMove(e){
	index = Math.floor((e.clientX-bar.getBoundingClientRect().left)/barW*data.timestamps);
	if(index < 0)
		index = 0;
	if(index >= data.timestamps)
		index = data.timestamps-1;
    updateData();
	updated=true;
}

function drawIndex (){
    //Timeline bar
	barC.fillStyle = "rgb(100,100,100)";
	barC.fillRect(0,0,barW,barH);
	barC.fillStyle = "rgb(0,150,150)";
	barC.fillRect(Math.floor(((barW-5)/(data.timestamps-1))*(index)),0,5,barH);
	
    //Text timer at the left of the bar
	timerC.fillStyle = "#333333";
	timerC.fillRect(0,0,timerW,barH);
	timerC.fillStyle="white";
	timerC.font = "10px serif";
	timerC.textBaseline = "top";
	timerC.fillText(data.currTime+"/"+data.simulationLength+" ms",10,0);
}

function setupMenuItems(){
    var ul;
    var subMenuItems = document.getElementsByClassName("subMenuItem");
    for(var i=0; i<subMenuItems.length; i++){
        subMenuItems[i].addEventListener("mouseenter",function(e){
            ul = this.getElementsByTagName("ul")[0];
            ul.style.left = this.getBoundingClientRect().width+"px";
            ul.style.top = (this.getBoundingClientRect().top-this.parentNode.getBoundingClientRect().top-4)+"px";
            });
    }
}

function updateDataScaling(v,factor){
    if(factor == 'A'){
        data.updateScalingFactor(v);
    }else if(factor == 'B'){
        data.updateOffsetFactor(v);
    }else if(factor == 'C'){
        data.updatePowerFactor(v);
    }
    r2D.drawMiniLegend();
    graph.drawAxes();
}

function updateCorrelationParams(v,param){
	if(param == "delay"){
		corr.setMaxDelay(v);
	}else if(param == "distance"){
		corr.setMaxDistance(v);
	}else if(param == "average"){
		corr.setAverageBinNumber(v);
	}else if(param == "log"){
		corr.useLog(v);
	}else if(param == "logVal"){
		corr.setLog(v);
	}
}

function updateDataType(){
    var value = document.getElementById("dataTypeSelect").value;
    var ph = document.getElementById("paramPlaceholder"),phe=document.getElementById("paramPlaceholderEnd"),newNodes=[];
    //Remove previous elements (if any)
    var nodesToRemove=[],currNode;
    currNode=ph.nextSibling;
    while(currNode != phe){
        ph.parentNode.removeChild(currNode);
        currNode=ph.nextSibling;
    }
    
    //Add new corresponding elements
    if(value == "binned"){
        newNodes[0] = document.createElement("li");
        newNodes[0].innerHTML = 'Number of bins along x <input id="xNumber" type="number" value="40">';
        newNodes[1] = document.createElement("li");
        newNodes[1].innerHTML = 'Number of bins along y <input id="yNumber" type="number" value="40">';
        newNodes[2] = document.createElement("li");
        newNodes[2].className = 'spacer'
        newNodes[3] = document.createElement("li");
        newNodes[3].className = 'paramListSubtitle';
        newNodes[3].innerHTML = 'Analog signal (LFP)';
        newNodes[4] = document.createElement("li");
        newNodes[4].innerHTML = 'Number of bins along x<input id="lfpx" type="number" value="10">';
        newNodes[5] = document.createElement("li");
        newNodes[5].innerHTML = 'Number of bins along y<input id="lfpy" type="number" value="10">';
        // newNodes[6] = document.createElement("li");
        // newNodes[6].innerHTML = "Filename of LFP data";
        // newNodes[7] = document.createElement("li");
        // newNodes[7].innerHTML = '<textarea id="lfpFile" rows="1" cols="100"></textarea>';
    }else if(value == "neuron"){
        newNodes[0] = document.createElement("li");
        newNodes[0].innerHTML = "Names of the neuron position files";
        newNodes[1] = document.createElement("li");
        newNodes[1].innerHTML = '<textarea id="posFiles" rows="1" cols="100"></textarea>';
    }
    for(var i = newNodes.length - 1; i >= 0; i--){
        ph.parentNode.insertBefore(newNodes[i], ph.nextSibling);
    }
    
}
