# README.md

# Unsupervised and Supervised Machine Learning with Autoencoders and Ridge Regression

This project implements a combination of unsupervised (autoencoder) and supervised (Ridge Regression) machine learning techniques to make accurate predictions on complex data, even with a minimal training dataset.

## Project Overview

The pipeline starts with data loading, followed by scaling the pre-training and training labels. The pre-training data is then used to train an Autoencoder model to reduce the input features from 1000 to 32, with an intermediate hidden layer of 750.

The Autoencoder functions in two ways:

1. **Dimensionality Reduction:** The Autoencoder reconstructs the input from the latent layer. The reconstruction error is computed and used for the backward pass, allowing the model to learn the important features about the molecule.

2. **Prediction:** In addition to reconstruction, the latent layer is used to predict the LUMO energy (Lowest Unoccupied Molecular Orbital), and a prediction error is computed. The combination of these two errors facilitates learning the features containing information about the electron density, which is essential for LUMO prediction.

The Autoencoder is trained over 20 epochs (with a batch size of 256 and learning rate of 0.0001) using the Adam optimizer. The validation loss is tracked with a test split of 1000 data points.

Following the unsupervised learning stage, the small training set is passed through the encoder to obtain feature embeddings. These features are then used in a Ridge Regression model to predict the HOMO-LUMO values (Highest Occupied Molecular Orbital - Lowest Unoccupied Molecular Orbital). This linear model is then used to predict the labels of the test set, ensuring the scaling is reversed.

## Dependencies

This project is implemented in Python and requires the following libraries:

- PyTorch
- NumPy
- Scikit-learn
- Pandas

These can be installed using pip:

```shell
pip install torch numpy scikit-learn pandas
```

## Usage

Make sure you have the necessary dataset in your working directory. Running the Python script will perform the pre-processing, train the Autoencoder, perform feature embedding with the encoder, train the Ridge Regression model, and generate predictions for the test set.

Note that the architecture, hyperparameters (such as learning rate, batch size, etc.), and scaling may need to be adjusted to better suit your specific dataset.