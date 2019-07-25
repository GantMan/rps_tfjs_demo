import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import {
  IMAGE_WIDTH,
  IMAGE_HEIGHT,
  NUM_CHANNELS,
  BATCH_SIZE,
  NUM_TRAIN_ELEMENTS,
  NUM_TEST_ELEMENTS
} from './constants'

export const train = (model, data, numEpochs = 10) => {
  const metrics = ['loss', 'acc', 'val_acc']
  const container = {
    name: 'Model Training',
    styles: { height: '1000px' }
  }
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics)
  tfvis.visor().setActiveTab('Visor')

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(NUM_TRAIN_ELEMENTS)
    return [
      d.xs.reshape([
        NUM_TRAIN_ELEMENTS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        NUM_CHANNELS
      ]),
      d.labels
    ]
  })

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(NUM_TEST_ELEMENTS)
    return [
      d.xs.reshape([
        NUM_TEST_ELEMENTS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        NUM_CHANNELS
      ]),
      d.labels
    ]
  })

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: numEpochs,
    shuffle: true,
    callbacks: fitCallbacks
  })
}
