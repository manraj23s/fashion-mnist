# Deep Learning & Attribution Modeling Report

This project explores two powerful deep learning architectures—**Convolutional Neural Networks (CNNs)** and **Autoencoders**—using the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. The Fashion-MNIST dataset consists of 60,000 training images and 10,000 test images of 28x28 grayscale clothing items, providing a robust benchmark for image classification and representation learning tasks.

---

## Approach

### 1. **Model 1: Autoencoder**

**Purpose:**  
Autoencoders are unsupervised neural networks used for dimensionality reduction, feature extraction, and image denoising. They consist of two parts:
- **Encoder:** Compresses the input data into a low-dimensional latent space.
- **Decoder:** Reconstructs the original data from the compressed latent representation.

**Implementation Highlights:**
- The autoencoder is built using convolutional layers in both encoder and decoder, leveraging spatial pattern recognition.
- The model is trained for 10 epochs to minimize mean squared error (MSE) between input and reconstructed images.
- Achieved a validation loss (MSE) of **0.0334**.
- Utility functions such as `get_triple` (to extract input images, latent representations, and decoded images) and `show_encodings` (to visualize the process) were developed.
- Latent representations were visualized using **t-SNE**, providing intuitive plots of the learned feature space.

**Result:**  
The autoencoder successfully learned meaningful feature representations of clothing items, capturing underlying data structures. These features are valuable for downstream tasks like clustering, visualization, or as inputs to other machine learning models for classification.

---

### 2. **Model 2: Convolutional Neural Network (CNN)**

**Purpose:**  
CNNs are designed specifically for image data and are widely used for classification tasks due to their ability to detect and hierarchically compose local patterns (such as edges, textures, and shapes).

**Implementation Highlights:**
- The CNN architecture includes convolutional layers, padding, batch normalization, flattening, dense layers, and dropout for regularization.
- Dataset is imported and split in the same manner as for the autoencoder, ensuring comparability.
- Model is trained for 10 epochs with an adaptive learning rate schedule, focusing on maximizing classification accuracy.

**Result:**  
The CNN achieved a **validation accuracy of 92%** within just 10 epochs, demonstrating its effectiveness for image classification in the Fashion-MNIST domain.

---

## Process Overview

1. **Data Preparation:**
   - Downloaded and preprocessed the Fashion-MNIST dataset.
   - Split the data into training and test sets.

2. **Model Development & Training:**
   - Built and trained an autoencoder to learn compressed latent representations.
   - Built and trained a CNN for image classification.

3. **Evaluation & Visualization:**
   - For the autoencoder: Examined reconstructed images and visualized latent spaces using t-SNE.
   - For the CNN: Evaluated accuracy and learning curves, confirming robust performance.

---

## Key Takeaways

- **CNNs** excel at image classification, leveraging convolutional layers to learn spatial hierarchies of features.
- **Autoencoders** provide powerful unsupervised feature learning and dimensionality reduction, making complex data more accessible and interpretable.
- Both models benefit from convolutional architectures when dealing with image data, as evidenced by their strong performance on the Fashion-MNIST dataset.

---

## Example Results

- **Autoencoder:**  
  - Validation Loss (MSE): **0.0334**  
  - Clear latent space visualizations and high-quality image reconstructions.
- **CNN:**  
  - Validation Accuracy: **92%** after 10 epochs.

---

## Future Work

- Further optimize architectures and hyperparameters for even better accuracy and reconstruction.
- Experiment with other unsupervised and semi-supervised learning techniques on the Fashion-MNIST dataset.
- Apply learned features from autoencoders to downstream tasks such as clustering or anomaly detection.

---

**Project by:** Manraj Singh  
*For questions or collaboration, please contact [manraj23singh@gmail.com](mailto:manraj23singh@gmail.com)*
