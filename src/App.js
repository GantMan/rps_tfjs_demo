import React, { Component } from 'react'
import logo from './logo.svg'
import gant from './corn.png'
import './App.css'
import { RPSDataset } from './tfjs/data.js'
import { getBetterModel, getSimpleModel } from './tfjs/models.js'
import { showAccuracy, showConfusion, showExamples } from './tfjs/evaluationHelpers.js'
import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

async function train(model, data) {
  const metrics = ['loss', 'acc', 'val_acc']
  const container = {
    name: 'Model Training',
    styles: { height: '1000px' }
  }
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics)

  const BATCH_SIZE = 512
  const TRAIN_DATA_SIZE = 2100
  const TEST_DATA_SIZE = 420

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE)
    return [d.xs.reshape([TRAIN_DATA_SIZE, 64, 64, 1]), d.labels]
  })

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE)
    return [d.xs.reshape([TEST_DATA_SIZE, 64, 64, 1]), d.labels]
  })

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 20,
    shuffle: true,
    callbacks: fitCallbacks
  })
}

class App extends Component {
  render() {
    return (
      <div className="App">
        <header className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
          <a
            className="App-link"
            href="https://infinite.red"
            target="_blank"
            rel="noopener noreferrer"
          >
            Infinite Red
          </a>
          <a
            className="App-link"
            href="http://gantlaborde.com/"
            target="_blank"
            rel="noopener noreferrer"
          >
            Gant Laborde
          </a>
          <hr />
        </header>
        <div class="Main">
          <p>
            We'll be working with a fun dataset for the classic game Rock Paper Scissors, provided 
            here: <a href="http://www.laurencemoroney.com/rock-paper-scissors-dataset/" target="_blank">Rock Paper Scissors Dataset</a>
          </p>
          <p>
            We'll show progress in the TensorflowJS Vis panel.  You'll see it when you click the load and show button below.
          </p>
          <button
            class="myButton"
            onClick={async () => {
              const data = new RPSDataset()
              this.data = data
              await data.load()
              await showExamples(data)
            }}
          >
            Load and Show Examples
          </button>

          <button
            class="myButton"
            onClick={async () => {
              const model = getBetterModel()
              tfvis.show.modelSummary({ name: 'Model Architecture' }, model)
              this.model = model
            }}
          >
            Create Model
          </button>

          <button
            class="myButton"
            onClick={async () => {
              await train(this.model, this.data)
            }}
          >
            Train
          </button>

          <button
            class="myButton"
            onClick={async () => {
              await showAccuracy(this.model, this.data)
              await showConfusion(this.model, this.data)
            }}
          >
            Check
          </button>
        </div>
        <img src={gant}/>
      </div>
    )
  }
}

export default App
