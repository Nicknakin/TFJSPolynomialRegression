//GLOBAL VARIABLES
var realXs = [];
var realYs = [];
var coefficients;
var learningRate = 0.1;
var dragging = false;
var dashed = 2;
var button;

const degree = 3;
const optimizer = tf.train.adam(learningRate);

//GLOBAL VARIABLES

function setup(){
	createCanvas(window.innerWidth,window.innerHeight);
	button = createButton("Clear");
	button.position(0, height+10);
	button.mousePressed(() => {realXs = []; realYs = [];});
	coefficients = new Array(degree+1);
	for(let i = 0; i < coefficients.length; i++){
		coefficients[i] = tf.variable(tf.scalar(Math.random()*2-1));
	}
}

function draw(){
	drawGraph();
	if(realXs.length > 1 && !dragging){
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
		color(255);
		stroke(255,128,0);
		strokeWeight(3);
		for(let i = dashed; i < theoryYs.length; i+= 2*dashed){
			line(theoryXs[i-dashed], theoryYs[i-dashed], theoryXs[i], theoryYs[i]);
		}
	}
}

function predict(receivedXs){
	receivedXs = mapArray(receivedXs, 0, width, -1, 1);
	let tensorXs = tf.tensor1d(receivedXs);
	let tensorYs = tf.zeros([receivedXs.length]).add(coefficients[0]);
	for(let i = 1; i < coefficients.length; i++){
			tensorYs = tensorYs.add(tensorXs.pow(tf.scalar(i)).mul(coefficients[i]));

	}
	return tensorYs;
}

function calcLoss(prediction){
	return tf.losses.meanSquaredError(tf.tensor1d(mapArray(realYs, 0, height, 1, -1)), prediction);
}

function optimize(){
	optimizer.minimize(() => calcLoss(predict(realXs)));
}

function drawGraph(){
	strokeWeight(1);
	background(0);
	stroke(255);
	line(width/2,0,width/2, height);
	line(0, height/2, width, height/2);
	fill(0,128,255);
	noStroke();
	for(let i = 0; i < realXs.length; i++){
		ellipse(realXs[i],realYs[i], 8);
	}

}

function mapArray(arr, min, max, newMin, newMax){
	let tmp = arr.slice();
	for(let i = 0; i < tmp.length; i++){
		tmp[i] = map(tmp[i], min, max, newMin, newMax);
	}
	return tmp;
}

function mousePressed(){
	dragging = true;
	if(mouseX >= 0 && mouseX < width && mouseY >= 0 && mouseY < height){
		realXs.push(mouseX);
		realYs.push(mouseY);
	}
}

function mouseReleased(){
	dragging = false;
}

function mouseDragged(){
	mousePressed();
}
