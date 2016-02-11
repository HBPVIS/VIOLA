if(Visu === undefined)
	var Visu = {};

Visu.Graph = function(panel,data){
    
	this.data = data;
	
	this.maxSpikes = 0;
    
    //var divTimeAxe = document.createElement("div");
    //divTimeAxe.id = "divTimeAxe";
    //divTimeAxe.style.marginLeft = "30px";
    
    var canvasTimeAxe = document.createElement("canvas");
	canvasTimeAxe.width = 500;
	canvasTimeAxe.height = 50;
    canvasTimeAxe.style.position = "absolute";
    canvasTimeAxe.style.left = "10px";
    canvasTimeAxe.style.top = "310px";
	canvasTimeAxe.id = "canvasTimeAxe";
    
    panel.insertBefore(canvasTimeAxe,panel.childNodes[0]);

    //var canvasTimeAxe
	
	var canvasAxe = document.createElement("canvas");
	canvasAxe.width = 30;
	canvasAxe.height = 300;
	canvasAxe.id = "canvasAxe";	
	var canvasGraph = document.createElement("canvas");
	canvasGraph.width = 400;
	canvasGraph.height = 300;
	canvasGraph.id = "canvasGraph";
	var canvasLegend = document.createElement("canvas");
	canvasLegend.width = 70;
	canvasLegend.height = 300;
	canvasLegend.id = "canvasLegend";
	
	this.panel = panel;
	panel.insertBefore(canvasLegend,panel.childNodes[0]);
	panel.insertBefore(canvasGraph,panel.childNodes[0]);
	panel.insertBefore(canvasAxe,panel.childNodes[0]);
	
	panel.ondragstart = function (e) {
		e.dataTransfer.setData("elemT","dragPanel");
		e.dataTransfer.setData("id",e.target.id);
		e.dataTransfer.setData("parent",e.target.parentNode);
		
		console.log("drag of "+e.target.id);
	};
	
	panel.ondragover = function (e) {
		e.preventDefault();
		return false;
	};
	
	panel.ondrop = function (e) {
		e.preventDefault();

		if(e.dataTransfer.getData("elemT")=="dragPanel"){
			var elem = document.getElementById(e.dataTransfer.getData("id"));
			var prevParent = elem.parentNode;
			this.parentNode.appendChild(elem);
			prevParent.appendChild(this);
		}
		
		
		return false;
	};
	
	this.ctxL = document.getElementById("canvasLegend").getContext("2d");
	this.ctx = document.getElementById("canvasGraph").getContext("2d");
	this.ctxA = document.getElementById("canvasAxe").getContext("2d");
    this.ctxT = document.getElementById("canvasTimeAxe").getContext("2d");
	
	this.svgW = 400;
	this.svgH = 300;

	this.timeOffset = 0;
	this.displayPop = [];
	
	for(var i=0; i<data.nLayers; i++){
		this.displayPop[i] = true;
	}

	this.spikesDataSize = this.svgW;	//Number of miliseconds displayed on graph (actually 1 per pixel)

	this.spikesData = [];
	
	this.total = [];
	
	for(var i = 0 ; i < data.nLayers ; i++){
		this.spikesData[i] = new Uint32Array(this.data.timestamps);
	}
	
	for(var i=0; i<this.data.timestamps; i++)
			this.total[i] = 0;
	
	this.setZoom(document.getElementById("zoomSlider").value);

	this.totalRatio = 0;
	
	this.bWidth = Math.ceil(this.svgW/this.spikesDataSize);
	this.eBWidth = Math.ceil(this.svgW/this.spikesDataSize);
}

Visu.Graph.prototype = {
	
	setZoom: function (z,index){
                // spikesDataSize is the percentage of timestamps
                this.spikesDataSize=Math.round(z * this.data.timestamps);
		this.bWidth = Math.ceil(this.svgW/this.spikesDataSize);
		this.eBWidth = this.svgW/this.spikesDataSize;

                document.getElementById("zoomValue").innerHTML=this.spikesDataSize * this.data.resolution;

        this.setOffset(index);
		this.displayData(index);
        this.drawTimeAxe();
	},

	//Put spike data into graph's data table "spikesData"
	readData: function (s){
		this.initTable(this.spikesData[s],this.data.timestamps);
		for(var k = 0 ; k < this.data.timestamps ; k++){
			for(var i = 0 ; i < this.data.xNeurons ; i++){
				for(var j = 0 ; j < this.data.yNeurons ; j++){
					this.spikesData[s][k] += this.data.datasets[s][k][i*this.data.yNeurons+j];
				}
			}
            this.spikesData[s][k] /= this.data.xNeurons*this.data.yNeurons;
		}
	},
	
	readNeuronData: function (s){
		console.log("reading neuron data "+s);
		var spikes;
		this.initTable(this.spikesData[s],this.data.timestamps);
		for(var k = 0 ; k < this.data.timestamps ; k++){
			spikes = this.data.neuronSpikes[s][k];
			if(spikes !== undefined){
				this.spikesData[s][k] = spikes.length;
			}   
		}			
	},

	initTable: function (table,size){
		for(var i=0; i<size; i++){
			table[i] = 0;
		}
	},

	//Compute total spikes to display relative values
	computeData: function (){
		
		this.maxSpikes = 0;
		
		//Compute total spikes
		for(var i=0; i<this.data.timestamps; i++){
			this.total[i] = 0;
			for(var j=0; j<this.data.nLayers; j++){
				if(this.data.mReady[j] && this.displayPop[j])
					this.total[i]+=this.spikesData[j][i];
			}
            
			if(this.total[i]>this.maxSpikes){
				this.maxSpikes=this.total[i];
			}
		}
        
        this.maxSpikes /= this.data.nLayers;
		
		this.totalRatio = this.svgH/2/this.maxSpikes/this.data.nLayers;
	},

	setOffset: function (index){
		//Move offset to center graph
		if(index < this.spikesDataSize/2){
			this.timeOffset = 0;
		}else if (index >= this.data.timestamps-this.spikesDataSize/2){
			this.timeOffset = this.data.timestamps-this.spikesDataSize;
		}else{
			this.timeOffset = index-this.spikesDataSize/2;
		}
	},
	
	disableDrag: function (v){
		if(v){
			this.panel.draggable=false;
		}else{
			this.panel.draggable=true;
		}
	},

	//Display graph
	displayData: function (index){
		this.ctx.fillStyle="black";
		this.ctx.fillRect(0,0,this.svgW,this.svgH);
		
		var tOffset;
		var offset;
		
		for(var i=0; i<this.spikesDataSize; i++){
			offset=0;
			tOffset = i+this.timeOffset;
			for(var j=0; j<this.data.nLayers; j++){
				if(this.data.mReady[j] && this.displayPop[j]){
					this.ctx.fillStyle = this.data.layerColors[j];
					this.ctx.fillRect(Math.floor(i*this.eBWidth),offset,
					this.bWidth,this.svgH*this.spikesData[j][tOffset]/this.total[tOffset]);
					offset+=this.svgH*this.spikesData[j][tOffset]/this.total[tOffset];
				}
			}
		}
		//Display total spikes amount	
		this.ctx.fillStyle="black";
		this.ctx.lineWidth=2;
		this.ctx.lineJoin="round";
		this.ctx.beginPath();
		this.ctx.moveTo(0, this.svgH-this.data.getScaledValue(this.total[0+this.timeOffset])*this.totalRatio);
		for(var i=0; i<this.spikesDataSize; i++){
			this.ctx.lineTo(i*this.eBWidth,this.svgH-this.data.getScaledValue(this.total[i+this.timeOffset])*this.totalRatio);
		}
		this.ctx.lineTo(this.svgW, this.svgH-this.data.getScaledValue(this.total[this.spikesDataSize-1+this.timeOffset])*this.totalRatio);
		this.ctx.stroke();

        //Simulation time display
		this.ctx.textBaseline = "top";
		this.ctx.font = "12px serif";
        this.ctx.fillRect(0,0,this.ctx.measureText(this.data.simulationLength+"/"+this.data.simulationLength+" ms").width+10,16);
        this.ctx.fillStyle="white";
		this.ctx.fillText(this.data.currTime+"/"+this.data.simulationLength+" ms",5+this.ctx.measureText(this.data.simulationLength+"/"+this.data.simulationLength+" ms").width - this.ctx.measureText(this.data.currTime+"/"+this.data.simulationLength+" ms").width,2);
        
        this.drawIndexGraph(index);		
		
	},

	togglePop: function (b,v){
		this.displayPop[v]=b;
		
		this.computeData();
		this.displayData();
		this.drawAxes();
	},

	drawIndexGraph: function (index){
		//Draw mean spikes canvas index
		this.ctx.fillStyle="white";
		this.ctx.fillRect(Math.floor((index-this.timeOffset)*(this.svgW)/this.spikesDataSize)-1,0,2,this.svgH);
	},

	drawLegend: function (){
		//Draw legend
		this.ctxL.font = "19px serif";
		this.ctxL.textBaseline = "middle";
		for(var i = 0 ; i < this.data.nLayers ; i++){
			this.ctxL.fillStyle = this.data.layerColors[i];
			this.ctxL.fillRect(10,i*this.svgH/this.data.nLayers,20,this.svgH/this.data.nLayers);
			this.ctxL.fillStyle="black";
			this.ctxL.fillText(this.data.layerNames[i], 34, i*this.svgH/this.data.nLayers+this.svgH/this.data.nLayers/2);
		}
	},
    
    drawTimeAxe: function (){

        this.ctxT.clearRect(0,0,500,50);
        
        this.ctxT.fillStyle = "black";
        
		this.ctxT.font = "12px serif";
		this.ctxT.textBaseline = "middle";
        
        this.ctxT.fillText("Delay (ms)",230-this.ctxT.measureText("Delay (ms)").width/2,30);
        
        this.ctxT.fillRect(229,0,2,8);
        this.ctxT.fillText("0",230-this.ctxT.measureText("0").width/2,15);

        // tick spacing rounded to
        var roundSpacing = Math.round(this.spikesDataSize.toPrecision(1) * 0.1);
        // maximum number of ticks (0 ms excluded, only positive direction)
        var maxNumTicks = 8;
        var realTicksSpacing = Math.ceil(this.spikesDataSize/(2*maxNumTicks*roundSpacing)) * roundSpacing
        // console.log("real ticks spacing : "+realTicksSpacing);
        var ticksSpacing = Math.round(400*realTicksSpacing/this.spikesDataSize);
        // console.log("ticks spacing : "+ticksSpacing);
        var numTicks = Math.floor(200/ticksSpacing);
        // console.log("ticks : "+numTicks);

        for(var i=1; i<numTicks+1; i++){
            this.ctxT.fillRect(230-ticksSpacing*i,0,2,5);
            this.ctxT.fillText("-"+i*realTicksSpacing*this.data.resolution,230-ticksSpacing*i-this.ctxT.measureText("-"+i*realTicksSpacing*this.data.resolution).width/2,15);
            this.ctxT.fillRect(228+ticksSpacing*i,0,2,5);
            this.ctxT.fillText(i*realTicksSpacing*this.data.resolution,229+ticksSpacing*i-this.ctxT.measureText(i*realTicksSpacing*this.data.resolution).width/2,15);
        }
    },
	
	drawAxes: function (){
        
        var sp = [Math.floor(this.data.getUnscaledValue(this.maxSpikes/this.data.resolution)),
            Math.floor(this.data.getUnscaledValue(3*this.maxSpikes/this.data.resolution/4)),
            Math.floor(this.data.getUnscaledValue(this.maxSpikes/this.data.resolution/2)),
            Math.floor(this.data.getUnscaledValue(this.maxSpikes/this.data.resolution/4))];
        var unit = ["","K","M","G","T"];
        var uToUse = [0,0,0,0];
        
        for(var i=0; i<4; i++){
            while(sp[i] >= 1000){
                sp[i]/=1000;
                uToUse[i]++;
            }
            if(sp[i] < 0){
                sp[i] = 0;
                uToUse[i] = 0;
            }else if(sp[i] < 10)
                sp[i] = Math.floor(100*sp[i])/100;
            else if(sp[i] < 100)
                sp[i] = Math.floor(10*sp[i])/10;
            else
                sp[i] = Math.floor(sp[i]);
        }
        
		this.ctxA.font = "10px serif";
		this.ctxA.textBaseline = "middle";
		// this.ctxA.fillStyle="white";
		this.ctxA.clearRect(0, 0, 30, this.svgH);
		this.ctxA.fillStyle="black";
		this.ctxA.fillRect(28, this.svgH/2, 2, this.svgH/2);
		this.ctxA.fillRect(26, this.svgH/2, 4, 1);
		this.ctxA.fillRect(26, 5*this.svgH/8, 4, 1);
		this.ctxA.fillRect(26, 3*this.svgH/4, 4, 1);
		this.ctxA.fillRect(26, 7*this.svgH/8, 4, 1);
		this.ctxA.fillText(sp[0]+unit[uToUse[0]], 13-this.ctxA.measureText(sp[0]+unit[uToUse[0]]).width/2, this.svgH/2);
		this.ctxA.fillText(sp[1]+unit[uToUse[1]], 13-this.ctxA.measureText(sp[1]+unit[uToUse[1]]).width/2, 5*this.svgH/8);
		this.ctxA.fillText(sp[2]+unit[uToUse[2]], 13-this.ctxA.measureText(sp[2]+unit[uToUse[2]]).width/2, 3*this.svgH/4);
		this.ctxA.fillText(sp[3]+unit[uToUse[3]], 13-this.ctxA.measureText(sp[3]+unit[uToUse[3]]).width/2, 7*this.svgH/8);
		this.ctxA.save();
		this.ctxA.rotate(-Math.PI/2);
		this.ctxA.translate(-this.svgH, 0);
		this.ctxA.textBaseline = "middle";
		this.ctxA.font = "12px serif";
		this.ctxA.fillText("Mean spike count (1/s)",3*this.svgH/4-this.ctxA.measureText("Mean spike count (1/s)").width/2,15);
		this.ctxA.restore();
	},
}
