#  Hand Sign Classification Using CNN

Ever wondered how a computer could learn to "read" sign language? This project is a small step in that direction: a Convolutional Neural Network (CNN) that looks at a 28×28 grayscale photo of a hand and tells you which letter of the American Sign Language (ASL) alphabet it's showing.

It's built with TensorFlow/Keras, trains in a few minutes, and reaches around **93% accuracy on unseen data**.

---

## Why this project?

The classic MNIST handwritten-digit dataset is a rite of passage for ML beginners, but honestly, it's gotten too easy. Modern models ace it, and some researchers joke that you can tell many digit pairs apart by checking a single pixel.

[**Sign Language MNIST**](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) is a drop-in replacement with the same CSV format, but a far more interesting real-world problem: understanding hand gestures. Same simplicity, higher stakes.

## About the dataset

- **Format:** just like the original MNIST. Each row in the CSV is one image: a `label` column plus 784 pixel values (28×28, grayscale, 0–255).
- **Task:** multi-class classification over **24 letters** of the alphabet.
- **Why no J or Z?** Those two letters need hand *motion*, and a single still image can't capture that. Because labels still run from 0 to 25, the model uses 26 output nodes, with two of them simply never appearing in the data.

## How it works

### 1. Preprocessing
- Split each CSV into pixel features and labels.
- Reshape every row into a `28 × 28 × 1` image.
- Normalize pixels from `[0, 255]` to `[0, 1]`. This makes training faster and more stable, and keeps activations from saturating.

### 2. The model

```
Input (28×28×1)
  → Conv2D(30, 5×5, ReLU) → MaxPooling(2×2)
  → Conv2D(15, 3×3, ReLU) → MaxPooling(2×2)
  → Dropout(0.2)
  → Flatten
  → Dense(128, ReLU) → Dense(50, ReLU)
  → Dense(26, Softmax)
```

In plain English:
- The **convolutional layers** are the eyes. Early filters pick up edges and corners, later ones start to recognize finger and knuckle shapes.
- **Max pooling** shrinks the image down to its strongest signals, so the model doesn't care if the hand is slightly off-center.
- **Dropout** randomly switches off 20% of neurons during training so the network can't just memorize the training set.
- The **dense layers** combine those features into a decision, and **softmax** turns it into a probability for each letter.

### 3. Training
- **Loss:** sparse categorical cross-entropy
- **Optimizer:** Adam
- **Metric:** accuracy
- The trained model is saved as `sign_language_mnist_cnn.h5`.

## Results

| Split      | Accuracy |
|------------|----------|
| Training   | ~99.4%   |
| Validation | ~93.2%   |

The gap between the two tells us there's still some overfitting. That's a good hint for where to improve next (see below).

The notebook also runs an end-to-end sanity check: it takes an image from the validation set, runs a prediction, and displays it alongside the predicted and actual labels.

## Getting started

### Requirements

```bash
pip install pandas numpy opencv-python tensorflow matplotlib
```

### Run it

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) and place `sign_mnist_train.csv` and `sign_mnist_test.csv` where the notebook expects them. The notebook uses `/content/`, which is the Google Colab default, so update the paths if you're running locally.
2. Open `HandsignClassificationUsingCNN.ipynb` in Jupyter or Google Colab.
3. Run all cells.

### Use the trained model

```python
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("sign_language_mnist_cnn.h5")

# image: a 28x28 grayscale array with values in 0-255
image = image.reshape(1, 28, 28, 1) / 255.0
predicted_class = np.argmax(model.predict(image))
```

The output is a class index from 0 to 25 (A=0, B=1, ... with 9 for J and 25 for Z unused).

## Project structure

```
├── HandsignClassificationUsingCNN.ipynb   # Full walkthrough: data → model → prediction
├── sign_language_mnist_cnn.h5             # Saved model (generated after training)
└── README.md
```

## Ideas for what's next

- 📷 **Real-time recognition** using a webcam feed
- 🔄 **Data augmentation** (rotations, zooms, shears) to handle different lighting and hand orientations and close the train/validation gap
- 📱 **Model compression** to run on phones and edge devices
- 🔤 **Word-level recognition** by chaining predictions together

## Acknowledgements

- Dataset: [Sign Language MNIST on Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- Built with [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/)

---

If you found this useful let me know. Feedback and pull requests are welcome!
