import React, { Component } from 'react'
import logo from './logo.svg'
import './App.css'
import { MnistData } from './data.js'
import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

function getModel() {
  const model = tf.sequential()

  const IMAGE_WIDTH = 28
  const IMAGE_HEIGHT = 28
  const IMAGE_CHANNELS = 1

  // In the first layer of out convolutional neural network we have
  // to specify the input shape. Then we specify some parameters for
  // the convolution operation that takes place in this layer.
  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    })
  )

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))

  // Repeat another conv2d + maxPooling stack.
  // Note that we have more filters in the convolution.
  model.add(
    tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    })
  )
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))

  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten())

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_OUTPUT_CLASSES = 10
  model.add(
    tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax'
    })
  )

  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  const optimizer = tf.train.adam()
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  })

  return model
}

async function train(model, data) {
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc']
  const container = {
    name: 'Model Training',
    styles: { height: '1000px' }
  }
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics)

  const BATCH_SIZE = 512
  const TRAIN_DATA_SIZE = 5500
  const TEST_DATA_SIZE = 1000

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE)
    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels]
  })

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE)
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels]
  })

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks
  })
}

async function showExamples(data) {
  // Create a container in the visor
  const surface = tfvis
    .visor()
    .surface({ name: 'Input Data Examples', tab: 'Input Data' })

  // Get the examples
  const examples = data.nextTestBatch(20)
  const numExamples = examples.xs.shape[0]

  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1])
    })

    const canvas = document.createElement('canvas')
    canvas.width = 28
    canvas.height = 28
    canvas.style = 'margin: 4px;'
    await tf.browser.toPixels(imageTensor, canvas)
    surface.drawArea.appendChild(canvas)

    imageTensor.dispose()
  }
}

const classNames = [
  'Zero',
  'One',
  'Two',
  'Three',
  'Four',
  'Five',
  'Six',
  'Seven',
  'Eight',
  'Nine'
]

function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28
  const IMAGE_HEIGHT = 28
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

async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data)
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds)
  const container = { name: 'Accuracy', tab: 'Evaluation' }
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames)

  labels.dispose()
}

async function showConfusion(model, data) {
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

class App extends Component {
  async componentDidMount() {
    // const data = new MnistData()
    // await data.load()
    // await showExamples(data)
    // train n stuff
    // const model = getModel()
    // tfvis.show.modelSummary({ name: 'Model Architecture' }, model)
    // await train(model, data)
    // show results
    // await showAccuracy(model, data)
    // await showConfusion(model, data)
  }

  render() {
    return (
      <div className="App">
        <header className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
          <a
            className="App-link"
            href="https://tfhub.dev/"
            target="_blank"
            rel="noopener noreferrer"
          >
            TFHub based Transfer Learning
          </a>
          <hr />
          <button
            onClick={async () => {
              const data = new MnistData()
              this.data = data
              await data.load()
              await showExamples(data)
            }}
          >
            Show Examples
          </button>

          <button
            onClick={async () => {
              const model = getModel()
              tfvis.show.modelSummary({ name: 'Model Architecture' }, model)
              this.model = model
            }}
          >
            Create Model
          </button>

          <button
            onClick={async () => {
              await train(this.model, this.data)
            }}
          >
            Train
          </button>

          <button
            onClick={async () => {
              await showAccuracy(this.model, this.data)
              await showConfusion(this.model, this.data)
            }}
          >
            Check
          </button>
        </header>
      </div>
    )
  }
}

export default App
