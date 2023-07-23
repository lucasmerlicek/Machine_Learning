# README.md

# Ridge Regression with Feature Transformation and Fine-Grained Hyperparameter Tuning

This project implements Ridge Regression, a powerful regularized linear regression variant, with feature transformations and fine-grained hyperparameter tuning.

## Project Overview

The pipeline begins with a transformation function that transforms the 5 input features into 21 new features using linear, quadratic, exponential, cosine, and constant transformations. This transformation process takes place by transforming the input matrix separately and then joining them together using numpy's hstack function.

The transformed dataset is then used to train a Ridge Regression model using Scikit-learn's Ridge class from the linear_model module. Ridge Regression utilizes a penalty for the L2 norm of the weight vector to avoid overfitting, which is controlled by the lambda term. This hyperparameter lambda is finely tuned by searching over a range of values to identify the optimal value.

The performance of the model for each lambda value is assessed using K-Fold Cross-Validation with 10 folds. The average Root Mean Squared Error (RMSE) is calculated for each lambda value.

The lambda value that yields the lowest average RMSE is chosen and used to compute the final weight vector. This final model is then trained on the entire dataset.

## Dependencies

This project is implemented in Python and requires the following libraries:

- NumPy
- Scikit-learn
- Pandas

These can be installed using pip:

```shell
pip install numpy scikit-learn pandas
```

## Usage

Ensure you have the necessary dataset in your working directory. Running the Python script will perform data transformations, train the Ridge Regression model, fine-tune the lambda hyperparameter, and generate the final model trained on the entire dataset.

Please note that the architecture and hyperparameters (such as lambda values, transformation types, etc.) may need to be adjusted to better suit your specific dataset.