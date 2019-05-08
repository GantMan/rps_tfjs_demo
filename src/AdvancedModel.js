import React from 'react'
import { TFWrapper } from './tfjs/evaluationHelpers'
import * as tf from '@tensorflow/tfjs'

const ADV_RPS_MODEL_URL = process.env.PUBLIC_URL + '/adv_rps/'
const ADV_LABELS_URL = ADV_RPS_MODEL_URL + 'labels.json'
const ADV_MODEL_JSON = ADV_RPS_MODEL_URL + 'model.json'

export default class AdvancedModel extends React.Component {
  videoRef = React.createRef()
  canvasRef = React.createRef()

  state = {
    loading: true
  }

  componentDidMount() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const webCamPromise = navigator.mediaDevices
        .getUserMedia({
          audio: false,
          video: {
            facingMode: 'user'
          }
        })
        .then(stream => {
          window.stream = stream
          this.videoRef.current.srcObject = stream
          return new Promise((resolve, _) => {
            this.videoRef.current.onloadedmetadata = () => {
              resolve()
            }
          })
        })

      const modelPromise = tf.loadGraphModel(ADV_MODEL_JSON)
      const labelsPromise = fetch(ADV_LABELS_URL).then(data => data.json())
      Promise.all([modelPromise, labelsPromise, webCamPromise])
        .then(values => {
          const [model, labels] = values
          this.setState({ loading: false })
          this.detectFrame(this.videoRef.current, model, labels)
        })
        .catch(error => {
          console.error(error)
        })
    }
  }

  componentWillUnmount = () => {
    // stop and collect garbage
    let stream = window.stream
    let tracks = stream.getTracks()

    tracks.forEach(track => {
      track.stop()
    })

    window.stream = null
  }

  detectFrame = (video, model, labels) => {
    TFWrapper(model)
      .detect(video)
      .then(predictions => {
        this.renderPredictions(predictions, labels)
        requestAnimationFrame(() => {
          // calm down when hidden!
          if (this.canvasRef.current) {
            this.detectFrame(video, model, labels)
          }
        })
      })
  }

  renderPredictions = (predictions, labels) => {
    if (this.canvasRef.current) {
      const ctx = this.canvasRef.current.getContext('2d')
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
      // Font options.
      const font = '16px sans-serif'
      ctx.font = font
      ctx.textBaseline = 'top'
      predictions.forEach(prediction => {
        const x = prediction.bbox[0]
        const y = prediction.bbox[1]
        const width = prediction.bbox[2]
        const height = prediction.bbox[3]
        const label = labels[parseInt(prediction.class)]
        // Draw the bounding box.
        ctx.strokeStyle = '#FF0000'
        ctx.lineWidth = 4
        ctx.strokeRect(x, y, width, height)
        // Draw the label background.
        ctx.fillStyle = '#FF0000'
        const textWidth = ctx.measureText(label).width
        const textHeight = parseInt(font, 10) // base 10
        ctx.fillRect(x, y, textWidth + 4, textHeight + 4)
      })

      predictions.forEach(prediction => {
        const x = prediction.bbox[0]
        const y = prediction.bbox[1]
        const label = labels[parseInt(prediction.class)]
        // Draw the text last to ensure it's on top.
        ctx.fillStyle = '#000000'
        ctx.fillText(label, x, y)
      })
    }
  }

  render() {
    return (
      <div className="advancedContainer">
        {this.state.loading && (
          <p id="advancedLoadText">Please wait, loading advanced model</p>
        )}
        <video
          className="advancedCam"
          autoPlay
          playsInline
          muted
          ref={this.videoRef}
          width="600"
          height="500"
        />
        <canvas
          className="advancedBox"
          ref={this.canvasRef}
          width="600"
          height="500"
        />
      </div>
    )
  }
}
