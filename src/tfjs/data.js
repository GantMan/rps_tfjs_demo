import * as tf from '@tensorflow/tfjs'

// 64x64
const IMAGE_SIZE = 4096
const NUM_CLASSES = 3
const NUM_DATASET_ELEMENTS = 2520

const TRAIN_TEST_RATIO = 5 / 6

const NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS)
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS

const RPS_IMAGES_SPRITE_PATH = 'http://localhost:3000/data.png'
const RPS_LABELS_PATH = 'http://localhost:3000/labels_uint8'

export class RPSDataset {
  constructor() {
    this.shuffledTrainIndex = 0
    this.shuffledTestIndex = 0
  }

  async load() {
    // Make a request for the RPS sprited image.
    const img = new Image()
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    const imgRequest = new Promise((resolve, _reject) => {
      img.crossOrigin = ''
      img.onload = () => {
        img.width = img.naturalWidth
        img.height = img.naturalHeight

        const datasetBytesBuffer = new ArrayBuffer(
          NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4
        )

        const chunkSize = 60
        canvas.width = img.width
        canvas.height = chunkSize

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
            datasetBytesBuffer,
            i * IMAGE_SIZE * chunkSize * 4,
            IMAGE_SIZE * chunkSize
          )
          ctx.drawImage(
            img,
            0,
            i * chunkSize,
            img.width,
            chunkSize,
            0,
            0,
            img.width,
            chunkSize
          )

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)

          for (let j = 0; j < imageData.data.length / 4; j++) {
            // currently just handling single channel
            // just read the red.
            datasetBytesView[j] = imageData.data[j * 4] / 255
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer)

        resolve()
      }
      img.src = RPS_IMAGES_SPRITE_PATH
    })

    const labelsRequest = fetch(RPS_LABELS_PATH)
    const [_imgResponse, labelsResponse] = await Promise.all([
      imgRequest,
      labelsRequest
    ])

    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer())

    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS)
    this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS)

    // Slice the the images and labels into train and test sets.
    // This style of slicing hopes that they have been RANDOMIZED BEFORE
    // they show up here.  Otherwise your test set might be all the same class
    // UGH!
    this.trainImages = this.datasetImages.slice(
      0,
      IMAGE_SIZE * NUM_TRAIN_ELEMENTS
    )
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS)
    this.trainLabels = this.datasetLabels.slice(
      0,
      NUM_CLASSES * NUM_TRAIN_ELEMENTS
    )
    this.testLabels = this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS)
  }

  nextTrainBatch(batchSize) {
    return this.nextBatch(
      batchSize,
      [this.trainImages, this.trainLabels],
      () => {
        this.shuffledTrainIndex =
          (this.shuffledTrainIndex + 1) % this.trainIndices.length
        return this.trainIndices[this.shuffledTrainIndex]
        // return this.shuffledTrainIndex // For debugging, no rando
      }
    )
  }

  nextTestBatch(batchSize) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      this.shuffledTestIndex =
        (this.shuffledTestIndex + 1) % this.testIndices.length
      return this.testIndices[this.shuffledTestIndex]
      // return this.shuffledTestIndex // For debugging, no rando
    })
  }

  nextBatch(batchSize, data, index) {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE)
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES)

    for (let i = 0; i < batchSize; i++) {
      const idx = index()

      const image = data[0].slice(
        idx * IMAGE_SIZE,
        idx * IMAGE_SIZE + IMAGE_SIZE
      )
      batchImagesArray.set(image, i * IMAGE_SIZE)

      const label = data[1].slice(
        idx * NUM_CLASSES,
        idx * NUM_CLASSES + NUM_CLASSES
      )
      batchLabelsArray.set(label, i * NUM_CLASSES)
    }

    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE])
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES])
    // console.log(batchLabelsArray)
    return { xs, labels }
  }
}
