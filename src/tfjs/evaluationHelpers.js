import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

const classNames = ['Rock', 'Paper', 'Scissors']

const IMAGE_WIDTH = 64
const IMAGE_HEIGHT = 64
const NUM_CHANNELS = 3

export const doSinglePrediction = async (model, img) => {
  // First get logits
  const logits = tf.tidy(() => {
    img = tf.browser.fromPixels(img)
    // Bring it down to gray
    const gray_mid = img.mean(2)
    const gray = gray_mid.expandDims(2) // back to (width, height, 1)
    let resized
    if (img.shape[0] !== IMAGE_WIDTH || img.shape[1] !== IMAGE_WIDTH) {
      const alignCorners = true
      resized = tf.image.resizeBilinear(
        gray,
        [IMAGE_WIDTH, IMAGE_HEIGHT],
        alignCorners
      )
    }

    // Singe-element batch
    const batched = resized.reshape([1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
    // return logits
    return model.predict(batched)
  })

  const values = await logits.data()
  // cleanup logits
  logits.dispose()
  // return class + prediction of all
  return classNames.map((className, idx) => ({
    className,
    probability: values[idx]
  }))
}

const doPrediction = (model, data, testDataSize = 420) => {
  const testData = data.nextTestBatch(testDataSize)
  const testxs = testData.xs.reshape([
    testDataSize,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    1
  ])
  const labels = testData.labels.argMax([-1])
  const preds = model.predict(testxs).argMax([-1])

  testxs.dispose()
  return [preds, labels]
}

export const showAccuracy = async (model, data, title = 'Accuracy') => {
  const [preds, labels] = doPrediction(model, data)
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds)
  const container = { name: title, tab: 'Evaluation' }
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames)
  tfvis.visor().setActiveTab('Evaluation')

  labels.dispose()
}

export const showConfusion = async (
  model,
  data,
  title = 'Confusion Matrix'
) => {
  const [preds, labels] = doPrediction(model, data)
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds)
  const container = { name: title, tab: 'Evaluation' }
  tfvis.render.confusionMatrix(container, {
    values: confusionMatrix,
    tickLabels: classNames
  })

  labels.dispose()
}

export const showExamples = async data => {
  // Create a container in the visor
  const surface = tfvis
    .visor()
    .surface({ name: 'RPS Data Examples', tab: 'Input Data' })

  // Get the examples
  const examples = data.nextTestBatch(42)

  tf.unstack(examples.xs).forEach(async tensor => {
    const imageTensor = tensor.reshape([
      IMAGE_WIDTH,
      IMAGE_HEIGHT,
      NUM_CHANNELS
    ])
    // Re-organize to be num_channels last
    const canvas = document.createElement('canvas')
    canvas.width = IMAGE_WIDTH
    canvas.height = IMAGE_HEIGHT
    canvas.style = 'margin: 4px;'
    await tf.browser.toPixels(imageTensor, canvas)
    surface.drawArea.appendChild(canvas)

    tensor.dispose()
    imageTensor.dispose()
  })
}
