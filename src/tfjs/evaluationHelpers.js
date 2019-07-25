import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS } from './constants'

const classNames = ['Rock', 'Paper', 'Scissors']

export const doSinglePrediction = async (model, img, options = {}) => {
  // First get input tensor
  const resized = tf.tidy(() => {
    img = tf.browser.fromPixels(img)
    if (NUM_CHANNELS === 1) {
      // Bring it down to gray
      const gray_mid = img.mean(2)
      img = gray_mid.expandDims(2) // back to (width, height, 1)
    }
    // assure (img.shape[0] === IMAGE_WIDTH && img.shape[1] === IMAGE_WIDTH
    const alignCorners = true
    return tf.image.resizeBilinear(
      img,
      [IMAGE_WIDTH, IMAGE_HEIGHT],
      alignCorners
    )
  })

  const logits = tf.tidy(() => {
    // Singe-element batch
    const batched = resized.reshape([
      1,
      IMAGE_WIDTH,
      IMAGE_HEIGHT,
      NUM_CHANNELS
    ])

    // return the logits
    return model.predict(batched)
  })

  const values = await logits.data()

  // if we want a visual
  const { feedbackCanvas } = options
  if (feedbackCanvas) {
    await tf.browser.toPixels(resized.div(tf.scalar(255)), feedbackCanvas)
  }
  // cleanup tensors
  resized.dispose()
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
    NUM_CHANNELS
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

// provided by https://github.com/cloud-annotations/object-detection-react
// trained via IBM cloud https://cloud-annotations.github.io/training/object-detection/cli/index.html
export const TFWrapper = model => {
  const calculateMaxScores = (scores, numBoxes, numClasses) => {
    const maxes = []
    const classes = []
    for (let i = 0; i < numBoxes; i++) {
      let max = Number.MIN_VALUE
      let index = -1
      for (let j = 0; j < numClasses; j++) {
        if (scores[i * numClasses + j] > max) {
          max = scores[i * numClasses + j]
          index = j
        }
      }
      maxes[i] = max
      classes[i] = index
    }
    return [maxes, classes]
  }

  const buildDetectedObjects = (
    width,
    height,
    boxes,
    scores,
    indexes,
    classes
  ) => {
    const count = indexes.length
    const objects = []
    for (let i = 0; i < count; i++) {
      const bbox = []
      for (let j = 0; j < 4; j++) {
        bbox[j] = boxes[indexes[i] * 4 + j]
      }
      const minY = bbox[0] * height
      const minX = bbox[1] * width
      const maxY = bbox[2] * height
      const maxX = bbox[3] * width
      bbox[0] = minX
      bbox[1] = minY
      bbox[2] = maxX - minX
      bbox[3] = maxY - minY
      objects.push({
        bbox: bbox,
        class: classes[indexes[i]],
        score: scores[indexes[i]]
      })
    }
    return objects
  }

  const detect = input => {
    const batched = tf.tidy(() => {
      const img = tf.browser.fromPixels(input)
      // Reshape to a single-element batch so we can pass it to executeAsync.
      return img.expandDims(0)
    })

    const height = batched.shape[1]
    const width = batched.shape[2]

    return model.executeAsync(batched).then(result => {
      const scores = result[0].dataSync()
      const boxes = result[1].dataSync()

      // clean the webgl tensors
      batched.dispose()
      tf.dispose(result)

      const [maxScores, classes] = calculateMaxScores(
        scores,
        result[0].shape[1],
        result[0].shape[2]
      )

      const prevBackend = tf.getBackend()
      // run post process in cpu
      tf.setBackend('cpu')
      const indexTensor = tf.tidy(() => {
        const boxes2 = tf.tensor2d(boxes, [
          result[1].shape[1],
          result[1].shape[3]
        ])
        return tf.image.nonMaxSuppression(
          boxes2,
          maxScores,
          20, // maxNumBoxes
          0.5, // iou_threshold
          0.5 // score_threshold
        )
      })
      const indexes = indexTensor.dataSync()
      indexTensor.dispose()
      // restore previous backend
      tf.setBackend(prevBackend)

      return buildDetectedObjects(
        width,
        height,
        boxes,
        maxScores,
        indexes,
        classes
      )
    })
  }
  return {
    detect: detect
  }
}
