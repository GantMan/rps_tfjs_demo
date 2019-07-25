import * as tf from '@tensorflow/tfjs'
import {
  IMAGE_SIZE,
  NUM_CLASSES,
  NUM_DATASET_ELEMENTS,
  NUM_CHANNELS,
  BYTES_PER_UINT8,
  NUM_TRAIN_ELEMENTS,
  NUM_TEST_ELEMENTS
} from './constants'

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
    // Load full dataset sprite
    const imgRequest = new Promise((resolve, _reject) => {
      img.crossOrigin = ''
      img.onload = () => {
        img.width = img.naturalWidth
        img.height = img.naturalHeight

        // Every possible pixel and value
        const datasetBytesBuffer = new ArrayBuffer(
          NUM_DATASET_ELEMENTS * IMAGE_SIZE * BYTES_PER_UINT8 * NUM_CHANNELS // * 4 because number of bytes
        )

        // Chunk size: ratio of Test set size (tweak as needed)
        const chunkSize = Math.floor(NUM_TEST_ELEMENTS * 0.15)
        canvas.width = img.width
        canvas.height = chunkSize

        // Read in images in chunkSize for speed
        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
            datasetBytesBuffer, // buffer
            i * chunkSize * IMAGE_SIZE * BYTES_PER_UINT8 * NUM_CHANNELS, // byteOffset * 4 because RGBA format
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
          let x = 0
          // Jumps by 4 storing RGBA
          for (let j = 0; j < imageData.data.length; j += 4) {
            // Stores R, then G, then B, then A
            for (let i = 0; i < NUM_CHANNELS; i++) {
              datasetBytesView[x++] = imageData.data[j + i] / 255
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
      IMAGE_SIZE * NUM_TRAIN_ELEMENTS * NUM_CHANNELS
    )
    this.testImages = this.datasetImages.slice(
      IMAGE_SIZE * NUM_TRAIN_ELEMENTS * NUM_CHANNELS
    )
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
    const xs = tf.tensor3d(batchImagesArray, [
      batchSize,
      IMAGE_SIZE,
      NUM_CHANNELS
    ])
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES])
    return { xs, labels }
  }
}
