import * as tf from '@tensorflow/tfjs'

const IMAGE_SIZE = 64 * 64
const NUM_CLASSES = 3
const NUM_DATASET_ELEMENTS = 2520
// 1-4 (Red+Green+Blue+Alpha)
const NUM_CHANNELS = 1

const TRAIN_TEST_RATIO = 5 / 6

const NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS)
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS

const RPS_IMAGES_SPRITE_PATH = '/data.png'
const RPS_LABELS_PATH = '/labels_uint8'

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

        // Chunk size cannot exceed test elements
        const chunkSize = Math.floor(NUM_TEST_ELEMENTS * 0.3)
        canvas.width = img.width
        canvas.height = chunkSize

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
            datasetBytesBuffer, // buffer
            i * IMAGE_SIZE * chunkSize * 4, // byteOffset * 4 because RGBA format
            IMAGE_SIZE * chunkSize * NUM_CHANNELS // length
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

          // RGBA of image pixels (0-255)
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)

          // assign all RGBA right on over (when using all 4)
          if (NUM_CHANNELS === 4) {
            datasetBytesView = imageData.data
          } else {
            for (let j = 0; j < imageData.data.length / 4; j++) {
              // red channel is imageData.data[j * 4] / 255
              // green channel is imageData.data[j * 4 + 1] / 255
              // etc.
              for (let x = 0; x < NUM_CHANNELS; x++) {
                datasetBytesView[j + x] = imageData.data[j * 4 + x] / 255
              }
            }
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
    // This style of slicing hopes that they have been randomized BEFORE
    // they show up here.  Otherwise your test set might be all the same class
    // UGH!  I guess double randomization is fine.
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
    const batchImagesArray = new Float32Array(
      batchSize * IMAGE_SIZE * NUM_CHANNELS
    )
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES)

    // Create a batchSize of images
    for (let i = 0; i < batchSize; i++) {
      const idx = index()

      const startPoint = idx * IMAGE_SIZE * NUM_CHANNELS
      const image = data[0].slice(
        startPoint,
        startPoint + IMAGE_SIZE * NUM_CHANNELS
      )
      batchImagesArray.set(image, i * IMAGE_SIZE * NUM_CHANNELS)

      const label = data[1].slice(
        idx * NUM_CLASSES,
        idx * NUM_CLASSES + NUM_CLASSES
      )
      batchLabelsArray.set(label, i * NUM_CLASSES)
    }

    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE])
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES])
    return { xs, labels }
  }
}
