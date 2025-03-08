# Handwritten Digit Classification using CNN

## Overview
This project classifies handwritten digits (0-9) using a **Convolutional Neural Network (CNN)**. It utilizes deep learning techniques to recognize digits from images, commonly used in digit recognition applications like postal code reading and bank check processing.

## Dataset
- **Source:** MNIST dataset
- **Features:** 28x28 grayscale images of handwritten digits
- **Target Variable:** Digit labels (0-9)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/digit-classification.git
   cd digit-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Model Architecture
1. **Convolutional Layers:** Extract spatial features from images.
2. **Pooling Layers:** Reduce dimensionality while retaining important features.
3. **Fully Connected Layers:** Classify digits based on extracted features.
4. **Activation Functions:** ReLU for hidden layers and Softmax for output.
5. **Loss Function & Optimizer:** Categorical Cross-Entropy loss with Adam optimizer.

## Training & Evaluation
- Model is trained using MNIST dataset.
- Accuracy, loss, and confusion matrix are used for evaluation.
- Achieves high accuracy in digit classification tasks.

## Usage
- Load the trained model:
  ```python
  from tensorflow.keras.models import load_model
  model = load_model('digit_classifier.h5')
  ```
- Make predictions:
  ```python
  import numpy as np
  sample_image = np.random.rand(1, 28, 28, 1)  # Example input
  predicted_digit = np.argmax(model.predict(sample_image))
  print(predicted_digit)
  ```

## Results & Improvements
- Achieved high accuracy on MNIST dataset.
- Possible improvements: Data augmentation, deeper networks, and transfer learning.
