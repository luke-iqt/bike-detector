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

async function app() {
    console.log('Loading mobilenet..');
  
    // Load the model.
    //const model = await tfjs-converter.loadGraphModel(MODEL_URL);
    const model = await tf.loadGraphModel(MODEL_URL);
    //net = await mobilenet.load();
    console.log('Successfully loaded model');
    
    // Create an object from Tensorflow.js data API which could capture image 
    // from the web camera as Tensor.
    const webcamConfig = { resizeWidth: 128, resizeHeight: 128, centerCrop: true}
    const webcam = await tf.data.webcam(webcamElement,webcamConfig);
    while (true) {
      const img = await webcam.capture();
      //const result = await net.classify(img);
      let offset = tf.scalar(127.5);
      let new_frame = img.sub(offset)
          .div(offset)
          .expandDims().reshape([-1, 128, 128, 3]);
      const result = await model.execute(new_frame);
  
      const bike = result.dataSync();
      document.getElementById('console').innerText = `
      Not a Bike: ${bike[0]} \n
      Bike: ${bike[1]} 
    `;
      // Dispose the tensor to release the memory.
      img.dispose();
  
      // Give some breathing room by waiting for the next animation frame to
      // fire.
      await tf.nextFrame();
    }
  }
app();