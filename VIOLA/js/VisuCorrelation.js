if(Visu === undefined)
	var Visu = {};

Visu.Correlation = function(panel,data){
    
	this.data = data;
	
	this.canvasW = 400;
	this.canvasH = 500;
    this.canvasColorH = 100;
	
	this.displayW = 300;
	this.displayH = 400;
	
	this.paddingW = (this.canvasW - this.displayW)/2;
	this.paddingH = (this.canvasH - this.displayH)/2;
	
	var canvas = document.createElement("canvas");
	canvas.width = this.canvasW;
	canvas.height = this.canvasH;
	canvas.id = "canvasCorrelation";
	
	this.panel = panel;
	
	panel.insertBefore(canvas,panel.childNodes[0]);
	
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
	
	this.ctx = document.getElementById("canvasCorrelation").getContext("2d");
	
	this.layer1 = 0;
	this.layer2 = 0;
	this.maxDelay = 20;
	this.maxDistance = 10;
	this.average_bins_num = 10;
	this.corrImg = [];
    
    //Data scaling
    this.minVal = 0;
    this.maxVal = 1;
	this.logScale = false;
	this.logVal = 10;
	
	//DataScalingLegend
	
}

Visu.Correlation.prototype = {

	computeData: function (){
		console.log("computing...");
		var tStart = new Date();
		
		this.layer1 = document.getElementById("popCorr1").value;
		this.layer2 = document.getElementById("popCorr2").value;
		
		var i,j,k,x,y,new_pos;
		var average_bins_x = [], average_bins_y = [];
		
		//Take 10 random bins
		for(i = 0; i < this.average_bins_num; i++){
			new_pos = false;
			
			while(!new_pos){
				x = Math.floor(Math.random()*this.data.xNeurons);
				y = Math.floor(Math.random()*this.data.yNeurons);
				
				//Check if pos taken
				new_pos = true;
				for(j = 0; j < i; j++){
					if(average_bins_x[i] == x && average_bins_y[i] == y){
						new_pos = false;
					}
				}
			}
			average_bins_x[i] = x;
			average_bins_y[i] = y;
		}
		
		//Erase previous images
		for(i = 0; i < 2*this.maxDelay*this.maxDistance; i++){
			this.corrImg[i] = 0;
		}
		
		//Compute new image
		var line = [],temp_line,effective_bins;
		for(i = 0; i < this.maxDistance; i++){
			for(j = 0,effective_bins=0; j < this.average_bins_num; j++){
				temp_line = this.data.computeCrossCorrelation(this.layer1,average_bins_x[j],average_bins_y[j],this.layer2,(average_bins_x[j]+i)%this.data.xNeurons,average_bins_y[j],this.maxDelay);
				for(k = 0; k < 2*this.maxDelay; k++){
					this.corrImg[k+i*2*this.maxDelay] += temp_line[k];
				}
			}
			for(k = 0; k < 2*this.maxDelay; k++){
				this.corrImg[k+i*2*this.maxDelay] /= this.average_bins_num;
			}
		}
		
		var tEnd = new Date();
		console.log("data processed in "+(tEnd.getTime()-tStart.getTime())+" ms");
		
		this.draw();
	},
	
	disableDrag: function (v){
		if(v){
			this.panel.draggable=false;
		}else{
			this.panel.draggable=true;
		}
	},
    
	colorRect: function (d){
        if(d < 0)
            return "rgb(0,0,0)";
		if(d < 0.33)
			return "rgb("+Math.round(3*d*255)+",0,0)";
		else if(d < 0.66)
			return "rgb(255,"+Math.round(255*(d-0.33)*3)+",0)";
		else if(d < 1)
			return "rgb(255,255,"+Math.round(255*(d-0.66)*3)+")";
        else
            return "rgb(255,255,255)";
	},
	
	setMinVal: function(v){
		this.minVal = parseFloat(v);
		if(this.minVal >= this.maxVal){
			this.minVal = this.maxVal - 0.01;
			document.getElementById("rangeCorrMin").value = this.minVal;
		}
		document.getElementById("corrScaleMinVal").innerHTML = this.minVal;
		this.draw();
	},
	
	setMaxVal: function(v){
		this.maxVal = parseFloat(v);
		if(this.maxVal <= this.minVal){
			this.maxVal = this.minVal + 0.01;
			document.getElementById("rangeCorrMax").value = this.maxVal;
		}
		document.getElementById("corrScaleMaxVal").innerHTML = this.maxVal;
		this.draw();
	},
    
    scaleValue: function(v){
        if(v > this.maxVal)
            return 1;
        else if (v < this.minVal)
            return 0;
        else
            return (v-this.minVal)/(this.maxVal-this.minVal);
    },

	draw: function (){
		this.ctx.clearRect(0,0,this.canvasW,this.canvasH);
		
		//Draw image into canvas
		var i,j,val;
		var xRatio = this.displayW/(2*this.maxDelay);
		var yRatio = this.displayH/(this.maxDistance);
		var s_xRatio = Math.ceil(this.displayW/(2*this.maxDelay));
		var s_yRatio = Math.ceil(this.displayH/(this.maxDistance));
		
		var log1 = Math.log(1+this.logVal);
		
		for(i = 0; i < 2*this.maxDelay; i++){
			for(j = 0; j < this.maxDistance; j++){
				val = this.corrImg[i+j*2*this.maxDelay];
				if(isNaN(val)){
					this.ctx.fillStyle = "rgb(0,0,0)";
				}else{
					if(this.logScale){
						this.ctx.fillStyle = this.colorRect(Math.log(1+this.logVal*this.scaleValue(val))/log1);
					}else{
						this.ctx.fillStyle = this.colorRect(this.scaleValue(val));
					}
				}
				this.ctx.fillRect(this.paddingW+i*xRatio,this.paddingH+j*yRatio,s_xRatio,s_yRatio);
			}
		}
		
		this.drawLegends();
	},
    
    drawLegends: function(){
		//Draw legends
		//Top (Title)
		this.ctx.fillStyle = "black";
		this.ctx.textBaseline = "middle";
		this.ctx.font = "20px serif";
		this.ctx.fillText("Cross correlation", this.canvasW/2-this.ctx.measureText("Cross correlation").width/2, this.paddingH/2);
		
		this.ctx.font = "12px serif";
		this.ctx.textBaseline = "middle";
		
		//Left (distance)
		var distance = this.maxDistance*this.data.xSize/this.data.xNeurons;
		var sc = [distance/4,
				distance/2,
				3*distance/4,
				distance];
				
		for(var i=0; i<4; i++){
			if(sc[i]<1)
				sc[i] = sc[i].toPrecision(1);
			else
				sc[i] = sc[i].toPrecision(2);
		}
				
		this.ctx.fillText(0, this.paddingW-this.ctx.measureText(0).width-2, this.paddingH);
		this.ctx.fillText(sc[0], this.paddingW-this.ctx.measureText(sc[0]).width-2, this.paddingH+this.displayH/4);
		this.ctx.fillText(sc[1], this.paddingW-this.ctx.measureText(sc[1]).width-2, this.paddingH+this.displayH/2);
		this.ctx.fillText(sc[2], this.paddingW-this.ctx.measureText(sc[2]).width-2, this.paddingH+3*this.displayH/4);
		this.ctx.fillText(sc[3], this.paddingW-this.ctx.measureText(sc[3]).width-2, this.paddingH+this.displayH);
		this.ctx.save();
		this.ctx.rotate(-Math.PI/2);
		this.ctx.translate(-this.canvasH, 0);
		this.ctx.textBaseline = "top";
		this.ctx.fillText("Distance (mm)",this.canvasH/2-this.ctx.measureText("Distance (mm)").width/2,0);
		this.ctx.restore();
		
		//Bottom
		this.ctx.textBaseline = "bottom";
		this.ctx.fillText("Delay (ms)", this.canvasW/2-this.ctx.measureText("Delay (ms)").width/2, this.canvasH-10);
		this.ctx.textBaseline = "top";
		var delay = this.maxDelay*this.data.resolution;
		var sc = [-delay,
				-delay/2,
				delay/2,
				delay];
		this.ctx.fillText(sc[0].toPrecision(2), this.paddingW-this.ctx.measureText(sc[0].toPrecision(2)).width/2, this.paddingH+this.displayH+5);
		this.ctx.fillText(sc[1].toPrecision(2), this.paddingW+this.displayW/4-this.ctx.measureText(sc[1].toPrecision(2)).width/2, this.paddingH+this.displayH+5);
		this.ctx.fillText(0, this.canvasW/2-this.ctx.measureText(0).width/2, this.paddingH+this.displayH+5);
		this.ctx.fillText(sc[2].toPrecision(2), this.paddingW+3*this.displayW/4-this.ctx.measureText(sc[2].toPrecision(2)).width/2, this.paddingH+this.displayH+5);
		this.ctx.fillText(sc[3].toPrecision(2), this.paddingW+this.displayW-this.ctx.measureText(sc[3].toPrecision(2)).width/2, this.paddingH+this.displayH+5);
        
        //Right (color range)
		var grd=this.ctx.createLinearGradient(0,this.displayH,0,0);
		grd.addColorStop(0,this.colorRect(0));
		grd.addColorStop(0.33,this.colorRect(0.33));
		grd.addColorStop(0.66,this.colorRect(0.66));
		grd.addColorStop(1,this.colorRect(1));
        this.ctx.fillStyle = grd;
        this.ctx.fillRect(this.paddingW+this.displayW,this.paddingH,10,this.displayH);
		
		//Right (numbers)
		this.ctx.fillStyle = "black";
		this.ctx.textBaseline = "middle";
		this.ctx.fillText(this.maxVal, this.displayW+this.paddingW+20-this.ctx.measureText(this.maxVal).width/2, this.paddingH);
		this.ctx.fillText(this.minVal, this.displayW+this.paddingW+20-this.ctx.measureText(this.minVal).width/2, this.paddingH+this.displayH);
		this.ctx.save();
		this.ctx.rotate(-Math.PI/2);
		this.ctx.translate(-this.canvasH, 0);
		this.ctx.fillText("Correlation value",this.canvasH/2-this.ctx.measureText("Correlation value").width/2,this.canvasW-this.paddingW/2);
		this.ctx.restore();
    },
	
	setAverageBinNumber: function(v){
		this.average_bins_num = parseInt(v);
	},
	
	setMaxDelay: function(v){
		this.maxDelay = parseInt(v);
	},
	
	setMaxDistance: function(v){
		this.maxDistance = parseInt(v);
	},
	
	useLog: function(v){
		this.logScale = v;
		this.draw();
	},
	
	setLog: function(v){
		this.logVal = parseFloat(v);
		this.draw();
	},
}