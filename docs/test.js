let net;
const MODEL_URL = 'vww_128_color_bicycle_005_web_model/model.json';

async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  //net = await mobilenet.load();
  const model = await tf.loadGraphModel(MODEL_URL);
  console.log('Successfully loaded model');

  // Make a prediction through the model on our image.
  const imgEl = document.getElementById('img');
  let tensor = tf.browser.fromPixels(imgEl)
		.resizeNearestNeighbor([128, 128])
        .toFloat();

    let offset = tf.scalar(127.5);
    let new_frame = tensor.sub(offset)
        .div(offset)
        .expandDims().reshape([-1, 128, 128, 3]);
  
        const result = await model.execute(new_frame);
        result.print(true);
        const bike = result.dataSync();
        document.getElementById('console').innerText = `
        Not a Bike: ${bike[0]} \n
        Bike: ${bike[1]} 
      `;
  console.log(result);
}

app();