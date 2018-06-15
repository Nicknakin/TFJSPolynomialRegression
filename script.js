//GLOBAL VARIABLES

var realXs = [];
var realYs = [];
const degree = 1;
var coefficients;
var learningRate = 0.145;
const optimizer = tf.train.adamax(learningRate);

//GLOBAL VARIABLES

function setup(){
	createCanvas(800,800);
	coefficients = new Array(degree+1);
	for(let i = 0; i < coefficients.length; i++){
		coefficients[i] = tf.variable(tf.scalar(Math.random()*2-1));
	}
}

function draw(){
	drawGraph();
	if(realXs.length > 1){
		//TRAIN
		tf.tidy(() => {
			optimize();
		});
		
		//DRAW Function
		let theoryXs = [];
		for(let i = 0; i < width; i++){
			theoryXs.push(i);
		}
		
		let theoryYs = mapArray(predict(theoryXs).dataSync(), -1, 1, height, 0);
		for(let i = 1; i < theoryYs.length; i++){
			line(theoryXs[i-1], theoryYs[i-1], theoryXs[i], theoryYs[i]);
		}
	}
}

function predict(realXs){
	realXs = mapArray(realXs, 0, width, -1, 1);
	let zeroes = new Array(realXs.length);
	for(let i = 0; i < zeroes.length; i++)
		zeroes[i] = 0;
	let tensorXs = tf.tensor1d(realXs);
	let tensorYs = tf.tensor1d(zeroes);
	for(let i = 0; i < coefficients.length; i++){
		tensorYs = tensorYs.add(tensorXs.pow(tf.scalar(i)).mul(coefficients[i]));
	}
	return tensorYs;
}

function calcLoss(prediction){
	return prediction.sub(tf.tensor1d(mapArray(realYs, 0, height, 1, -1))).square().mean();
}

function optimize(){
	optimizer.minimize(() => calcLoss(predict(realXs)));
}

function drawGraph(){
	background(255);
	fill(0);
	noStroke();
	for(let i = 0; i < realXs.length; i++){
		ellipse(realXs[i],realYs[i], 8);
	}
	stroke(1);
	line(width/2,0,width/2, height);
	line(0, height/2, width, height/2);
	
}

function mapArray(arr, min, max, newMin, newMax){
	let tmp = arr.slice();
	for(let i = 0; i < tmp.length; i++){
		tmp[i] = map(tmp[i], min, max, newMin, newMax);
	}
	return tmp;
}

function mouseClicked(){
	realXs.push(mouseX);
	realYs.push(mouseY);
}
