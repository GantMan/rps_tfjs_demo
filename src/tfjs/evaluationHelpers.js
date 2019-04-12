import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

const classNames = [
    'Rock',
    'Paper',
    'Scissors'
  ]

  const IMAGE_WIDTH = 64
  const IMAGE_HEIGHT = 64
  
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
  
  export const showAccuracy = async (model, data) => {
    const [preds, labels] = doPrediction(model, data)
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds)
    const container = { name: 'Accuracy', tab: 'Evaluation' }
    tfvis.show.perClassAccuracy(container, classAccuracy, classNames)
  
    labels.dispose()
  }
  
  export const showConfusion = async (model, data) => {
    const [preds, labels] = doPrediction(model, data)
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds)
    const container = { name: 'Confusion Matrix', tab: 'Evaluation' }
    tfvis.render.confusionMatrix(
      container,
      { values: confusionMatrix },
      classNames
    )
  
    labels.dispose()
  }

  export const showExamples = async (data) => {
    // Create a container in the visor
    const surface = tfvis
      .visor()
      .surface({ name: 'RPS Data Examples', tab: 'Input Data' })
  
    // Get the examples
    const examples = data.nextTestBatch(42)
    const numExamples = examples.xs.shape[0]
  
    // Create a canvas element to render each example
    for (let i = 0; i < numExamples; i++) {
      const imageTensor = tf.tidy(() => {
        // Reshape the image to 64x64 px
        return examples.xs
          .slice([i, 0], [1, examples.xs.shape[1]])
          .reshape([64, 64, 1])
      })
  
      const canvas = document.createElement('canvas')
      canvas.width = 64
      canvas.height = 64
      canvas.style = 'margin: 4px;'
      await tf.browser.toPixels(imageTensor, canvas)
      surface.drawArea.appendChild(canvas)
  
      imageTensor.dispose()
    }
  }