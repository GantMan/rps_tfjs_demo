import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

const NUM_CLASSES = 3

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

export const trainLocals = (model, data, numEpochs = 20) => {
  const metrics = ['loss', 'acc', 'val_acc']
  const container = {
    name: 'Local Model Training',
    styles: { height: '1000px' }
  }
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics)
  tfvis.visor().setActiveTab('Visor')

  const [tensorImages, answers] = data
  const TRAIN_DATA_SIZE = answers.length / NUM_CLASSES
  const trainXs = tensorImages.reshape([TRAIN_DATA_SIZE, 64, 64, 1])
  const trainYs = tf.tensor2d(answers, [TRAIN_DATA_SIZE, NUM_CLASSES])

  return model.fit(trainXs, trainYs, {
    batchSize: TRAIN_DATA_SIZE,
    epochs: numEpochs,
    shuffle: true,
    callbacks: fitCallbacks
  })
}
