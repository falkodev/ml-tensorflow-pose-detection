const MODEL_PATH = '/movenet'
const IMG = document.getElementById('img')

let model = undefined

async function loadModel() {
  model = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true })

  const tensorInput = tf.zeros([1, 192, 192, 3], 'int32')
  const imgTensor = tf.browser.fromPixels(IMG)
  console.log('imgTensor.shape ====> ', imgTensor.shape)

  const cropStartPoint = [15, 170, 0]
  const cropSize = [345, 345, 3]
  const croppedTensor = tf.slice(imgTensor, cropStartPoint, cropSize)

  const resizedTensor = tf.image.resizeBilinear(croppedTensor, [192, 192], true).toInt()
  console.log('resizedTensor ====> ', resizedTensor.shape)

  const tensorOutput = model.predict(tf.expandDims(resizedTensor))
  const arrayOutput = await tensorOutput.array()

  console.log('arrayOutput ====> ', arrayOutput)
  tensorInput.dispose()
  imgTensor.dispose()
  croppedTensor.dispose()
  resizedTensor.dispose()
  tensorOutput.dispose()
  arrayOutput.dispose()
  model.dispose()
}

loadModel()