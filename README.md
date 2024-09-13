### Handwritten Digits Classification using Neural Networks

This project demonstrates the classification of handwritten digits (from the MNIST dataset) using two approaches: Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN). The project aims to compare the performance of these two methods in terms of accuracy and effectiveness for digit classification.

#### Dataset
The MNIST dataset, which consists of 70,000 images of handwritten digits (60,000 training samples and 10,000 test samples), was used for training and evaluating the models. Each image is 28x28 pixels and represents a grayscale value between 0 and 255. The dataset is a benchmark in the field of computer vision and machine learning.

#### Project Structure
The project is divided into two main parts:
1. **Artificial Neural Network (ANN) Implementation**
2. **Convolutional Neural Network (CNN) Implementation**

#### 1. Artificial Neural Network (ANN) Implementation
- **Flattening the Dataset:** 
  The 28x28 pixel images are flattened into a vector of size 784 (28 * 28) to be used as inputs for the neural network.

- **Basic Model (No Hidden Layers):**
  A very simple neural network with no hidden layers was first implemented to understand the baseline performance of the ANN. The model consists of:
  - **Input Layer:** 784 neurons
  - **Output Layer:** 10 neurons with a sigmoid activation function (to classify digits 0-9)

  ```python
  model = keras.Sequential([
      keras.layers.Input(shape=(784,)),
      keras.layers.Dense(10, activation='sigmoid')
  ])
  ```

  The model was compiled using the Adam optimizer, sparse categorical crossentropy as the loss function, and accuracy as the evaluation metric.

- **ANN with Hidden Layer:**
  To improve performance, an additional hidden layer with 100 neurons and a ReLU activation function was added. The new architecture is:
  - **Input Layer:** 784 neurons
  - **Hidden Layer:** 100 neurons, ReLU activation
  - **Output Layer:** 10 neurons, sigmoid activation

  ```python
  model = keras.Sequential([
      keras.layers.Input(shape=(784,)),
      keras.layers.Dense(100, activation='relu'),
      keras.layers.Dense(10, activation='sigmoid')
  ])
  ```

  Results:
  - Accuracy on the test dataset: **~97.5%**

#### 2. Convolutional Neural Network (CNN) Implementation
- **Why CNN?**
  Convolutional Neural Networks (CNNs) are more effective for image classification as they can capture spatial hierarchies in images using convolutional layers. This allows CNNs to detect patterns such as edges and shapes, which are critical in recognizing handwritten digits.

- **Model Architecture:**
  The CNN model consists of:
  - **Convolutional Layer:** 32 filters of size 3x3, ReLU activation
  - **MaxPooling Layer:** Pooling size of 2x2 to reduce spatial dimensions
  - **Flatten Layer:** To flatten the feature maps into a vector
  - **Dense Layers:** A fully connected hidden layer with 100 neurons and a ReLU activation function, followed by an output layer with 10 neurons and softmax activation for classification.

  ```python
  cnn = keras.Sequential([
      keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Flatten(),
      keras.layers.Dense(100, activation='relu'),
      keras.layers.Dense(10, activation='softmax')
  ])
  ```

  Results:
  - Accuracy on the test dataset: **~98.6%**

#### Performance Comparison
- The simple ANN without any hidden layers achieved around **91% accuracy** on the test set.
- The ANN with one hidden layer improved the performance significantly, achieving around **97.5% accuracy**.
- The CNN outperformed both ANN models, achieving **98.6% accuracy** due to its ability to extract spatial features from the image.

#### Confusion Matrix and Visualization
To further evaluate the model performance, confusion matrices were generated for both the ANN and CNN models, visualizing how well the models classified each digit. The confusion matrix provides insights into which digits the models tend to confuse, helping to analyze the performance more thoroughly.

- The confusion matrices were visualized using Seaborn's heatmap functionality.

#### Conclusion
This project highlights the effectiveness of CNNs over traditional ANNs for image classification tasks like handwritten digit recognition. While the ANN with hidden layers performed well, the CNN achieved superior accuracy due to its ability to capture and process spatial patterns in the images.

Here’s the updated "How to Run" section with the provided repository and file links:

#### How to Run
1. Install the required dependencies:
   ```bash
   pip install tensorflow matplotlib seaborn
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/Akmech001/Hand-Wriiten-Digits-Classification-using-Neural-Network.git
   cd Hand-Wriiten-Digits-Classification-using-Neural-Network
   ```
3. Open and run the Python notebook file:
   - If you are using Jupyter Notebook or JupyterLab:
     ```bash
     jupyter notebook digits_recognition_neural_network.ipynb
     ```
   Alternatively, you can view the notebook directly on GitHub [here](https://github.com/Akmech001/Hand-Wriiten-Digits-Classification-using-Neural-Network/blob/Data-Science-Capstone-Porject/digits_recognition_neural_network.ipynb).

By following these steps, you'll be able to train and test the models and compare the results for handwritten digit classification.
