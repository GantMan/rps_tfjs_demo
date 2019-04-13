import React, { Component } from 'react'
import gant from './corn.png'
import './App.css'
import { RPSDataset } from './tfjs/data.js'
import { getAdvancedModel, getSimpleModel } from './tfjs/models.js'
import {
  showAccuracy,
  showConfusion,
  showExamples
} from './tfjs/evaluationHelpers.js'
import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

async function train(model, data, numEpochs = 10) {
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
    epochs: numEpochs,
    shuffle: true,
    callbacks: fitCallbacks
  })
}

class App extends Component {
  state = {
    currentModel: null
  }

  render() {
    return (
      <div className="App">
        <header className="App-header">
          <h2>Rock Paper Scissors</h2>
          <h3>Machine Learning in the browser with TFJS</h3>
          <img src="./rps_circle.png" className="App-logo" alt="logo" />
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
        </header>
        <div class="Main">
          <p>
            We'll be working with a fun dataset for the classic game Rock Paper
            Scissors, provided here:{' '}
            <a
              href="http://www.laurencemoroney.com/rock-paper-scissors-dataset/"
              target="_blank"
            >
              Rock Paper Scissors Dataset
            </a>
          </p>
          <img src="./rps.jpg" alt="Rock Paper Scissors dataset" />
          <p>
            We'll show progress in the TensorflowJS Vis panel. You'll see it
            when you click the load and show button below.
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
          <p>PLACEHOLDER</p>
          <div class="GroupUp">
            <button
              class="myButton"
              onClick={async () => {
                this.setState({ currentModel: 'Simple' })
                const model = getSimpleModel()
                tfvis.show.modelSummary(
                  { name: 'Simple Model Architecture' },
                  model
                )
                this.model = model
              }}
            >
              Create Simple Model
            </button>
            <button
              class="myButton"
              onClick={async () => {
                this.setState({ currentModel: 'Advanced' })
                const model = getAdvancedModel()
                tfvis.show.modelSummary(
                  { name: 'Advanced Model Architecture' },
                  model
                )
                this.model = model
              }}
            >
              Create Advanced Model
            </button>
          </div>
          <p>PLACEHOLDER</p>
          <button
            class="myButton"
            onClick={async () => {
              await showAccuracy(this.model, this.data)
              await showConfusion(this.model, this.data, 'Untrained Matrix')
            }}
          >
            Check Untrained Model Results
          </button>
          <p>Train your Model</p>
          <button
            class="myButton"
            onClick={async () => {
              const numEpochs = this.state.currentModel === 'Simple' ? 12 : 20
              await train(this.model, this.data, numEpochs)
            }}
          >
            Train Your {this.state.currentModel} Model
          </button>
          <p>PLACEHOLDER</p>
          <button
            class="myButton"
            onClick={async () => {
              await showAccuracy(this.model, this.data)
              await showConfusion(this.model, this.data)
            }}
          >
            Check Model After Training
          </button>
        </div>
        <img src={gant} class="wiggle" />
      </div>
    )
  }
}

export default App
