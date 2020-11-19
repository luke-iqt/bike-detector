const webcamElement = document.getElementById('webcam');
//import * as tf from '@tensorflow/tfjs';
//import {loadGraphModel} from '@tensorflow/tfjs-converter';
//import data from './vww_128_color_bicycle_005_web_model/model.json';
const MODEL_URL = 'vww_128_color_bicycle_005_web_model/model.json';

//console.log(data);

//const model = await loadGraphModel(MODEL_URL);
//const cat = document.getElementById('cat');
//model.execute(tf.browser.fromPixels(cat));
let net;
var webcam;
var model;
let fps = 4;
let frame_delay = 1000 / fps;
var now = new Date();
var next_frame = now.getTime() + frame_delay;

async function setup() {
    console.log('Loading mobilenet..');
  
    // Load the model.
    //const model = await tfjs-converter.loadGraphModel(MODEL_URL);
    model = await tf.loadGraphModel(MODEL_URL);
    //net = await mobilenet.load();
    console.log('Successfully loaded model');
    
    // Create an object from Tensorflow.js data API which could capture image 
    // from the web camera as Tensor.
    const webcamConfig = { resizeWidth: 128, resizeHeight: 128, centerCrop: true, facingMode: 'environment'}
    webcam = await tf.data.webcam(webcamElement,webcamConfig);

}
async function runOnce() {
 
      const img = await webcam.capture();
      //const result = await net.classify(img);
      let offset = tf.scalar(127.5);
      let new_frame = img.sub(offset)
          .div(offset)
          .expandDims().reshape([-1, 128, 128, 3]);
      const result = await model.execute(new_frame);
  
      const predictions = result.dataSync();
      const bike = Math.floor(predictions[1]*100);
      const notBike = Math.floor(predictions[0]*100);
      document.getElementById('console').innerText = `
      Bike: ${bike}%
      Not a Bike: ${notBike}%
    `;
      // Dispose the tensor to release the memory.
      img.dispose();

      // Give some breathing room by waiting for the next animation frame to
      // fire.
      await tf.nextFrame();
      now = new Date();
      var nap_time = next_frame - now.getTime();
      next_frame = now.getTime() + frame_delay;  
      //console.log(nap_time);
      setTimeout(runOnce, nap_time);
  }
  async function app() {
await setup();

    now = new Date();
next_frame = now.getTime() + frame_delay;
runOnce();
  }

  app();