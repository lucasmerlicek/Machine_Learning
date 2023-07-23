# README.md

# Food Image Classification Using Pre-Trained ResNet50 Neural Network

This project leverages a powerful pre-trained neural network, ResNet50, to classify food images. The ResNet50 neural network is employed to extract image features, represented by the last hidden layer's output during a forward pass. These features, termed embeddings, are stored externally for future use.

## Project Overview

In the project's pipeline, all the images are initially transformed and embedded in batch sizes of 64. This dataset of embeddings is used to train another custom neural network.

The custom neural network comprises 6 fully-connected layers with 6144, 2048, 1024, 512, 256, and 128 units respectively. It concludes with a single output unit for binary classification. The input layer takes the embeddings of a triplet stacked horizontally. To build a robust network, a dropout regularization technique with p=0.4 is applied to all nodes except the input and output layers.

The training process involves splitting the training data into a training set (90%) and a validation set (10%). The training set is batched (64 per batch) and passed through the model. After each forward pass, the gradients are reset, and backpropagation is performed based on the computed loss. The loss used here is Binary Cross Entropy with Logits Loss. After each training iteration, the model is evaluated using the validation set, and the validation loss and accuracy are printed. The entire process is repeated for 13 epochs.

Once the model is trained, it is used on the test triplets to generate a result file with the predicted labels.

## Dependencies

This project is implemented in Python and requires the following libraries:

- PyTorch
- torchvision
- numpy
- pandas

These can be installed using pip:

```
pip install torch torchvision numpy pandas
```

## Usage

Ensure the necessary food image dataset and the corresponding embeddings file are in your working directory. Running the Python script will perform image transformations, train the custom neural network, and generate a result file with predicted labels.

Remember to adjust the architecture of the custom neural network and training hyperparameters (such as epochs, dropout probability, etc.) to better suit your specific dataset.

## Note

The original image database included 10,000 images and is not available on GitHub due to its size. Likewise, the final embeddings file is excluded. Be sure to generate these resources as necessary for your specific application.