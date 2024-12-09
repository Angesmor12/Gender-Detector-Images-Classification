
let allow = 1
let loadingImage = document.querySelector(".loading-image-container")
let path = "model_alexnet.onnx"
let width = 224
let height = 224
let channels = 3
let rgb = 1
let nor = 1

document.querySelector(".algorithm-input-test").addEventListener("change", (e)=>{
  path = e.target.value
  width = parseInt(e.target.selectedOptions[0].dataset.width)
  height = parseInt(e.target.selectedOptions[0].dataset.width)
  channels = parseInt(e.target.selectedOptions[0].dataset.n)
  rgb = parseInt(e.target.selectedOptions[0].dataset.rgb)
  nor = parseInt(e.target.selectedOptions[0].dataset.nor)
})

const predictionText = document.querySelector(".prediction-value")
const predictionContainer = document.querySelector(".prediction-container")

async function predict(inputFeatures,path, key) {

    const session = await ort.InferenceSession.create(path);

    const input = new Float32Array(inputFeatures);
    const tensor = new ort.Tensor('float32', input, [1, channels, width, height]);

    const feeds = {};
    feeds[key] = tensor

    const result = await session.run(feeds);

    const logits = result.output.data; 
    const probabilities = softmax(logits);

    return probabilities;
}

const uploadArea = document.querySelector('.upload-area');

uploadArea.addEventListener('dragover', (event) => {
    event.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (event) => {
    event.preventDefault();
    uploadArea.classList.remove('dragover');

    const files = event.dataTransfer.files;
    if (files.length) {
        handleFiles(files);
    }
});

uploadArea.addEventListener('click', () => {
  const input = document.createElement('input');
  input.type = 'file';
  input.multiple = true; 
  input.click();

  input.addEventListener('change', (event) => {
      const files = event.target.files;
      if (files.length) {
          handleFiles(files);
      }
  });
});

function softmax(logits) {
  const maxLogit = Math.max(...logits); 
  const expLogits = logits.map(logit => Math.exp(logit - maxLogit));
  const sumExp = expLogits.reduce((a, b) => a + b, 0);
  return expLogits.map(value => value / sumExp);
}

function normalizeImages(data, nor) {
  const rChannel = [];
  const gChannel = [];
  const bChannel = [];

  let mean = []
  let std = []

  if (nor == 1)
  {
  mean = [0.485, 0.456, 0.406];
  std = [0.229, 0.224, 0.225];}
  else {
  mean = [0.5, 0.5, 0.5];
  std = [0.5, 0.5, 0.5];
  }

  for (let i = 0; i < data.length; i += 4) {
      rChannel.push((data[i] / 255 - mean[0]) / std[0]);     
      gChannel.push((data[i + 1] / 255 - mean[1]) / std[1]);
      bChannel.push((data[i + 2] / 255 - mean[2]) / std[2]);
  }

  return [rChannel, gChannel, bChannel];
}

function normalizeGrayscale(data) {
  const grayChannel = [];
  const mean = 0.5; 
  const std = 0.5; 

  for (let i = 0; i < data.length; i += 4) {
    const grayscaleValue = (data[i] + data[i + 1] + data[i + 2]) / (3 * 255);
    grayChannel.push((grayscaleValue - mean) / std);
  }

  return grayChannel;
}

// File management function
function handleFiles(files) {
  [...files].forEach((file) => {

      loadingImage.classList.remove("hidden")
      predictionContainer.classList.add("invisible")
      
      // Validate the file format
      if (file.type != "image/jpeg" && file.type != "image/jpg" && file.type != "image/png") {
          loadingImage.classList.add("hidden")
          return window.alert("The file must be an image in one of the following formats: PNG, JPEG, or JPG.");
      }

      const reader = new FileReader();

      reader.onload = function (event) {
        const img = new Image();
    
        img.onload = async function () {
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            canvas.width = width;
            canvas.height = height;
    
            ctx.drawImage(img, 0, 0, width, height);
    
            const imageData = ctx.getImageData(0, 0, width, height);
            const data = imageData.data;

            let normalizedData = 0

            if (rgb == 1){

              const [rChannel, gChannel, bChannel] = normalizeImages(data, nor);
      
              normalizedData = new Float32Array(3 * width * height);
              normalizedData.set(rChannel, 0);
              normalizedData.set(gChannel, width * height);
              normalizedData.set(bChannel, 2 * width * height);}
            else {
              const grayChannel = normalizeGrayscale(data);
              normalizedData = new Float32Array(width * height);
              normalizedData.set(grayChannel, 0);
            }
            
            const predictedValue = await predict(normalizedData, "./models/" + path, "input");

            showPrediction(predictedValue)
        };
        img.src = event.target.result;

      };
      reader.readAsDataURL(file);

  });
}

function showPrediction(value){

  const classValue = value.indexOf(Math.max(...value))
  loadingImage.classList.add("hidden")
  predictionContainer.classList.remove("invisible")

  if (classValue == 0){
      predictionText.innerHTML = "Female"
  }
  else {
      predictionText.innerHTML = "Male"
  }
}

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
let attempt = 1

async function captureFrame() {

  if (allow == 1 && attempt == 1){

  attempt = 0;

  loadingImage.classList.remove("hidden")
  predictionContainer.classList.add("invisible")

  const context = canvas.getContext('2d');

  canvas.width = width;
  canvas.height = height;

  context.drawImage(video, 0, 0, width, height);

  const rawData = context.getImageData(0, 0, width, height);
  const data = rawData.data;

  let normalizedData = 0

  if (rgb == 1){

    const [rChannel, gChannel, bChannel] = normalizeImages(data, nor);

    normalizedData = new Float32Array(3 * width * height);
    normalizedData.set(rChannel, 0);
    normalizedData.set(gChannel, width * height);
    normalizedData.set(bChannel, 2 * width * height);}
  else {
    const grayChannel = normalizeGrayscale(data);
    normalizedData = new Float32Array(width * height);
    normalizedData.set(grayChannel, 0);
  }
  

  const predictedValue = await predict(normalizedData, "./models/" + path, "input");

  showPrediction(predictedValue)  
  
  attempt = 1;

  await new Promise(resolve => setTimeout(resolve, 3000));
  requestAnimationFrame(captureFrame);
}
}

const modalBody = document.querySelector(".md")
const modalBody2 = document.querySelector(".md2")

document.querySelector(".option-b1 button").addEventListener("click", ()=>{
  allow = 0
  modalBody.classList.remove("hidden")
  modalBody2.classList.add("hidden")
})

document.querySelector(".option-b2 button").addEventListener("click", ()=>{

  navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
    video.srcObject = stream;
            })
    .catch((err) => {
       console.error("Error accessing the camera: ", err);
      window.alert("Could not access the camera.");
  });

  modalBody.classList.add("hidden")
  modalBody2.classList.remove("hidden")
  allow = 1
  captureFrame();
})