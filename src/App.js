import React, { Component } from "react";
import Webcam from "react-webcam";
import gant from "./corn.png";
import "./Buttons.css";
import "./App.css";
import { RPSDataset } from "./tfjs/data.js";
import { getAdvancedModel, getSimpleModel } from "./tfjs/models.js";
import { train } from "./tfjs/train.js";
import {
  showAccuracy,
  showConfusion,
  showExamples,
  doSinglePrediction
} from "./tfjs/evaluationHelpers.js";
import AdvancedModel from "./AdvancedModel.js";
import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from "@tensorflow/tfjs";

// Slide down code
import { SlideDown } from "react-slidedown";
import "react-slidedown/lib/slidedown.css";

// Code Display Stuff
import AceEditor from "react-ace";
import "ace-builds/src-noconflict/mode-javascript";
import "ace-builds/src-noconflict/theme-monokai";

const DETECTION_PERIOD = 2000;

class App extends Component {
  state = {
    currentModel: null,
    webcamActive: false,
    camMessage: "",
    advancedDemo: false,
    loadDataMessage: "Load and Show Examples",
    code1: true,
    code2: true,
    code3: true,
    code4: true
  };

  _renderAdvancedModel = () => {
    if (this.state.advancedDemo) {
      return (
        <div>
          <AdvancedModel key="advancedDemo" />
          <p>Turn off ad-block where applicable</p>
        </div>
      );
    }
  };

  componentDidMount() {
    /*
    Some code for debugging, sorrrrryyyyyy where is the best place for this?
    */
    window.tf = tf;
  }

  _renderWebcam = () => {
    if (this.state.webcamActive) {
      return (
        <div className="results">
          <div>64x64 Input</div>
          <canvas id="compVision" />
          <div>{this.state.camMessage}</div>
          <Webcam ref={this._refWeb} className="captureCam" />
        </div>
      );
    }
  };

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  detectWebcam = async () => {
    await this.sleep(100);
    const video = document.querySelectorAll(".captureCam");
    const feedbackCanvas = document.getElementById("compVision");
    // assure video is still shown
    if (video[0]) {
      const options = { feedbackCanvas };
      const predictions = await doSinglePrediction(
        this.model,
        video[0],
        options
      );
      const camMessage = predictions
        .map(p => ` ${p.className}: %${(p.probability * 100).toFixed(2)}`)
        .toString();
      this.setState({ camMessage });
      setTimeout(this.detectWebcam, DETECTION_PERIOD);
    }
  };

  _refWeb = webcam => {
    this.webcam = webcam;
  };

  render() {
    return (
      <div className="App">
        <header className="App-header">
          <h2>Rock Paper Scissors</h2>
          <h3>Machine Learning in the browser with TensorFlow.js</h3>
          <img
            src="./rps_circle.png"
            className="App-logo"
            alt="logo"
            id="logo"
          />
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
        <div className="Main">
          <p>
            We'll be working with a fun dataset for the classic game, "Rock
            Paper Scissors", provided here:{" "}
            <a
              href="http://www.laurencemoroney.com/rock-paper-scissors-dataset/"
              target="_blank"
              rel="noopener noreferrer"
            >
              Rock Paper Scissors Dataset
            </a>
          </p>
          <img src="./rps.jpg" alt="Rock Paper Scissors dataset" />
          <p>
            We'll show progress in the TensorFlow.js Vis panel. You'll see it
            when you click the load and show button below. Press{" "}
            <span className="cod">`</span> or <span className="cod">~</span> key
            to hide this menu.
          </p>
          <div className="GroupUp">
            <button
              className="btn-3d blue"
              onClick={async () => {
                this.setState({ loadDataMessage: "Loading 10MB Data" });
                const data = new RPSDataset();
                this.data = data;
                await data.load();
                await showExamples(data);
                this.setState({ loadDataMessage: "Data Loaded!" });
              }}
            >
              {this.state.loadDataMessage}
            </button>
            <button
              className="btn-3d green"
              onClick={() => this.setState({ code1: !this.state.code1 })}
            >
              <strong>&lt;/&gt;</strong>
            </button>
          </div>
          <SlideDown closed={this.state.code1}>
            <AceEditor
              placeholder="Don't Edit"
              mode="javascript"
              theme="monokai"
              maxLInes={8}
              height="16em"
              name="load"
              fontSize={18}
              showPrintMargin={false}
              showGutter={true}
              readOnly={true}
              highlightActiveLine={false}
              value={`
// Use custom data object
const data = new RPSDataset();
// Store on object
this.data = data;
// Parse tensors into memory
await data.load();
// Use VIS to make sure it worked!
await showExamples(data);`}
              setOptions={{
                enableBasicAutocompletion: false,
                enableLiveAutocompletion: false,
                enableSnippets: false,
                showLineNumbers: false,
                tabSize: 2
              }}
            />
          </SlideDown>

          <p>
            Each of the examples have been loaded now. Due to this being a
            browser, the data is loaded with one{" "}
            <a href="./data.png" target="_blank" rel="noopener noreferrer">
              sprite-sheet
            </a>{" "}
            to get around sandboxing. My code to create sprite-sheets is
            available with{" "}
            <a
              href="https://github.com/GantMan/rps_tfjs_demo"
              target="_blank"
              rel="noopener noreferrer"
            >
              this repo on GitHub
            </a>
            .
          </p>
          <p>
            You now create the structure for the data, that hopefully works
            best.{" "}
            <strong>
              In this situation, an advanced model is a bad choice.
            </strong>{" "}
            An advanced model will train slower while overfitting this small and
            simple training data.
          </p>
          <div className="GroupUp">
            <button
              className={
                this.state.currentModel === "Simple"
                  ? "btn-3d blue activeModel"
                  : "btn-3d blue"
              }
              onClick={async () => {
                this.setState({ currentModel: "Simple" });
                const model = getSimpleModel();
                tfvis.show.modelSummary(
                  { name: "Simple Model Architecture" },
                  model
                );
                this.model = model;
              }}
            >
              Create Simple Model
            </button>
            <button
              className="btn-3d green"
              onClick={() => this.setState({ code2: !this.state.code2 })}
            >
              <strong>&lt;/&gt;</strong>
            </button>
          </div>
          <SlideDown closed={this.state.code2} style={{ width: "100%" }}>
            <AceEditor
              placeholder="Don't Edit"
              mode="javascript"
              theme="monokai"
              width="100%"
              name="load"
              fontSize={18}
              showPrintMargin={false}
              showGutter={true}
              readOnly={true}
              highlightActiveLine={false}
              value={`
export const getSimpleModel = () => {
  const model = tf.sequential()

  // In the first layer of out convolutional neural network we have
  // to specify the input shape. Then we specify some parameters for
  // the convolution operation that takes place in this layer.
  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS],
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

  // Our last layer is a dense layer which has 3 output units, one for each
  // output class (i.e. 0, 1, 2).
  const NUM_OUTPUT_CLASSES = 3
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
`}
              setOptions={{
                enableBasicAutocompletion: false,
                enableLiveAutocompletion: false,
                enableSnippets: false,
                showLineNumbers: false,
                tabSize: 2
              }}
            />
          </SlideDown>
          <p>OR</p>
          <div className="GroupUp">
            <button
              className={
                this.state.currentModel === "Advanced"
                  ? "btn-3d blue activeModel"
                  : "btn-3d blue"
              }
              onClick={async () => {
                this.setState({ currentModel: "Advanced" });
                const model = getAdvancedModel();
                tfvis.show.modelSummary(
                  { name: "Advanced Model Architecture" },
                  model
                );
                this.model = model;
              }}
            >
              Create Advanced Model
            </button>
            <button
              className="btn-3d green"
              onClick={() => this.setState({ code3: !this.state.code3 })}
            >
              <strong>&lt;/&gt;</strong>
            </button>
          </div>
          <SlideDown closed={this.state.code3} style={{ width: "100%" }}>
            <AceEditor
              placeholder="Don't Edit"
              mode="javascript"
              theme="monokai"
              name="load"
              width="100%"
              fontSize={18}
              showPrintMargin={false}
              showGutter={true}
              readOnly={true}
              highlightActiveLine={false}
              value={`
export const getAdvancedModel = () => {
  const model = tf.sequential()

  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS],
      kernelSize: 3,
      padding: 'same',
      filters: 32,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    })
  )

  // Downsample, batchnorm, and dropout!
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))
  model.add(tf.layers.batchNormalization())
  model.add(tf.layers.dropout({ rate: 0.25 }))

  model.add(
    tf.layers.conv2d({
      kernelSize: 3,
      filters: 64,
      padding: 'same',
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    })
  )
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))
  model.add(tf.layers.batchNormalization())
  model.add(tf.layers.dropout({ rate: 0.25 }))

  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer.
  model.add(tf.layers.flatten())

  // complex dense intermediate
  model.add(
    tf.layers.dense({
      units: 512,
      kernelRegularizer: 'l1l2',
      activation: 'relu'
    })
  )

  // Our last layer is a dense layer which has 3 output units, one for each
  // output class (i.e. 0, 1, 2).
  const NUM_OUTPUT_CLASSES = 3
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
`}
              setOptions={{
                enableBasicAutocompletion: false,
                enableLiveAutocompletion: false,
                enableSnippets: false,
                showLineNumbers: false,
                tabSize: 2
              }}
            />
          </SlideDown>
          <p>
            Creating a model, is the structure and blueprint. It starts off able
            to, but terrible at predicting.
          </p>
          <div className="GroupUp">
            <button
              className="btn-3d blue"
              onClick={async () => {
                // stop errors
                if (!this.data) return;
                if (!this.model) return;
                await showAccuracy(this.model, this.data);
                await showConfusion(this.model, this.data, "Untrained Matrix");
              }}
            >
              Check Untrained Model Results
            </button>
          </div>
          <p>
            Train your Model with your training data. In this case 2100 labeled
            images, over and over... but not <em>toooooo much.</em>
          </p>
          <div className="GroupUp">
            <button
              className="btn-3d blue"
              onClick={async () => {
                // stop errors
                if (!this.data) return;
                if (!this.model) return;
                const numEpochs =
                  this.state.currentModel === "Simple" ? 12 : 20;
                await train(this.model, this.data, numEpochs);
              }}
            >
              Train Your {this.state.currentModel} Model
            </button>
            <button
              className="btn-3d green"
              onClick={() => this.setState({ code4: !this.state.code4 })}
            >
              <strong>&lt;/&gt;</strong>
            </button>
          </div>
          <SlideDown closed={this.state.code4} style={{ width: "100%" }}>
            <AceEditor
              placeholder="Don't Edit"
              mode="javascript"
              theme="monokai"
              width="100%"
              name="load"
              fontSize={18}
              showPrintMargin={false}
              showGutter={true}
              readOnly={true}
              highlightActiveLine={false}
              value={`
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
`}
              setOptions={{
                enableBasicAutocompletion: false,
                enableLiveAutocompletion: false,
                enableSnippets: false,
                showLineNumbers: false,
                tabSize: 2
              }}
            />
          </SlideDown>
          <p>
            Now that our model has seen some stuff{" "}
            <span role="img" aria-label="woah">
              😳
            </span>
            <hr />
            It should be smarter at identifying RPS! We can now test it with 420
            RPS images it's never seen before.
          </p>
          <button
            className="btn-3d blue"
            onClick={async () => {
              // stop errors
              if (!this.data) return;
              if (!this.model) return;
              await showAccuracy(this.model, this.data, "Trained Accuracy");
              await showConfusion(
                this.model,
                this.data,
                "Trained Confusion Matrix"
              );
            }}
          >
            Check Model After Training
          </button>
          <p>
            We can now save our trained model! We can store it via downloading
            it, uploading it, or place the results in localstorage for access of
            the browser.
          </p>
          <p>
            The simple model size comes out to about 48Kb, but some models can
            be as large as 20+MBs! It depends how simple you keep the model. If
            you want the model trained above, you get two files by{" "}
            <a
              className="pointy"
              onClick={async () => {
                if (!this.model) return;
                await this.model.save("downloads://rps-model");
              }}
            >
              clicking here
            </a>
            . The <span className="cod">model.json</span> file demonstrates the
            structure of the model, and the weights are our non-random trained
            values that make the model accurate.
          </p>
          <h3>Now let's see if we can test our model with the real world!</h3>
          <img src="./rps_webcam_big.jpg" className="demo" alt="webcam demo" />
          <p>
            Keep in mind, the training data for this model had no background,
            and the model itself isn't practiced in dealing with noise and
            rotation. A more advanced model would do better, but for this demo
            you shouldn't have any problems getting consistent and accurate
            results. When testing on a webcam, you'll need to make the images as
            clean as you can. Every few seconds your webcam image will be
            converted to a 64x64 grayscale image for your model to classify.
          </p>
          <button
            className="btn-3d blue"
            onClick={async () => {
              // stop errors
              if (!this.model) return;
              this.setState(
                prevState => ({
                  advancedDemo: false,
                  webcamActive: !prevState.webcamActive,
                  camMessage: ""
                }),
                this.detectWebcam
              );
            }}
          >
            {this.state.webcamActive ? "Turn Webcam Off" : "Launch Webcam"}
          </button>
          {this._renderWebcam()}
          <p>
            Did our model work for you? Maybe it did, and maybe it didn't! It's
            a very simple model that we've created on very simple data. Don't
            feel bad if it didn't work.
          </p>
          <p>
            What does it look like to train a far more advanced model for hours
            that results in a 20+MB model? Here's an opportunity for you to try
            it yourself! This model isn't as diverse, but for demo purposes it's
            inspiring!
          </p>
          <button
            className="btn-3d blue"
            onClick={() => {
              this.setState(prevState => ({
                webcamActive: false,
                advancedDemo: !prevState.advancedDemo
              }));
            }}
          >
            {this.state.advancedDemo
              ? "Turn Off Advanced Demo"
              : "Show Advanced Demo"}
          </button>
          {this._renderAdvancedModel()}
          <p>
            Machine Learning is exciting! And now you're part of it, as you
            trained a model right in your browser. We've only scratched the
            surface of what you can build. Automating with ML on computers is
            only limited by our imagination!
          </p>
          <p>
            If you'd like to see more applications of TensorFlow.js be sure to
            check out{" "}
            <a
              href="https://nsfwjs.com"
              target="_blank"
              rel="noopener noreferrer"
            >
              NSFWJS.com
            </a>
            , or the very useful{" "}
            <a
              href="https://nicornot.com"
              target="_blank"
              rel="noopener noreferrer"
            >
              NicOrNot.com
            </a>
            . For more entertaining applications of Machine Learning, be sure to
            subscribe to our{" "}
            <a
              href="https://ai-fyi.com"
              target="_blank"
              rel="noopener noreferrer"
            >
              Newsletter
            </a>{" "}
            or follow my Fun Machine Learning Twitter account:
          </p>
          <a
            href="https://twitter.com/FunMachineLearn"
            target="_blank"
            rel="noopener noreferrer"
          >
            <img
              src="fml.png"
              alt="Fun Machine Learn Logo"
              style={{ width: "50%", marginLeft: "25%" }}
            />
            <p style={{ textAlign: "center" }}>@FunMachineLearn</p>
          </a>
        </div>
        <div className="GroupUp">
          <p className="outro">
            Follow me (Gant Laborde) and Infinite Red for cool new experiments,
            and let us know what cool things you've come up with.{" "}
            <em>
              We can help, we're available for AI consulting and{" "}
              <a
                href="https://academy.infinite.red/"
                target="_blank"
                rel="noopener noreferrer"
              >
                training
              </a>
              .
            </em>
          </p>
        </div>
        <div className="GroupUp">
          <img src={gant} className="wiggle me" alt="Gant Laborde" />
          <ul id="footer">
            <li>
              Website:{" "}
              <a
                href="http://gantlaborde.com"
                target="_blank"
                rel="noopener noreferrer"
              >
                GantLaborde.com
              </a>
            </li>
            <li>
              Twitter:{" "}
              <a
                href="https://twitter.com/gantlaborde"
                target="_blank"
                rel="noopener noreferrer"
              >
                @GantLaborde
              </a>
            </li>
            <li>
              Medium:{" "}
              <a
                href="https://medium.freecodecamp.org/@gantlaborde"
                target="_blank"
                rel="noopener noreferrer"
              >
                GantLaborde
              </a>
            </li>
            <li>
              ML Twitter:{" "}
              <a
                href="https://twitter.com/FunMachineLearn"
                target="_blank"
                rel="noopener noreferrer"
              >
                FunMachineLearn
              </a>
            </li>
            <li>
              GitHub:{" "}
              <a
                href="https://github.com/GantMan/rps_tfjs_demo"
                target="_blank"
                rel="noopener noreferrer"
              >
                RPS TFJS Demo
              </a>
            </li>
            <li>
              Newsletter:{" "}
              <a
                href="https://ai-fyi.com"
                target="_blank"
                rel="noopener noreferrer"
              >
                AI-FYI.com
              </a>
            </li>
            <li>
              <a
                href="https://infinite.red"
                target="_blank"
                rel="noopener noreferrer"
              >
                <img src="./ir.svg" id="InfiniteRed" alt="Infinite Red" />
              </a>
            </li>
          </ul>
        </div>
        <div className="GroupUp">
          <img src="./ml.png" id="closer" alt="RPS" />
          <h4>powered by</h4>
          <img
            src="./TF_FullColor_Horizontal.png"
            id="closer"
            alt="Tensorflow logo"
            style={{ paddingLeft: "-40px" }}
          />
        </div>
      </div>
    );
  }
}

export default App;
