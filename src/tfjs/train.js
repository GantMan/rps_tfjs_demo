import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

export const train = (model, data, numEpochs = 10) => {
  const metrics = ['loss', 'acc', 'val_acc']
  const container = {
    name: 'Model Training',
    styles: { height: '1000px' }
  }
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics)
  tfvis.visor().setActiveTab('Visor')

  const BATCH_SIZE = 512
  const TRAIN_DATA_SIZE = 2100
  const TEST_DATA_SIZE = 420
  const NUM_CHANNELS = 3
  const IMAGE_WIDTH = 64
  const IMAGE_HEIGHT = 64

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE)
    return [
      d.xs.reshape([TRAIN_DATA_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS]),
      d.labels
    ]
  })

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE)
    return [
      d.xs.reshape([TEST_DATA_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS]),
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
