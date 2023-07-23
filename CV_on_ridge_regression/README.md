# README.md

# Ridge Regression with Cross-Validation

This project demonstrates the implementation of Ridge Regression using the Scikit-learn library in Python. Ridge Regression is a linear regression variant enhanced with an L2 regularization term to prevent overfitting.

## Project Overview

Ridge Regression aims to find optimal weights that minimize the distance between actual and predicted y-values, considering a regularization term. In this implementation, we utilize the `linear_model.Ridge` class from Scikit-learn, where the lambda (`lam`) hyperparameter controls the regularization term's influence.

The intercept for this model is explicitly set to be at 0, implying that the weight vector's length is fixed.

To evaluate model performance, the code computes the Root Mean Squared Error (RMSE) between the real y-values and predicted y-values for a given test set. The `mean_squared_error` function from Scikit-learn's metrics module is used to calculate the RMSE.

The project also employs K-Fold Cross-Validation, a popular resampling technique used to estimate unseen data's model performance. The KFold class from Scikit-learn is used to split the dataset into ten equally-sized folds. Then, one of these folds is held out for testing, while the remaining nine are used for training. This process is performed ten times, and the average RMSE is computed for each lambda value.

## Dependencies

To run this project, the following Python libraries are required:

- Scikit-learn
- NumPy
- pandas

You can install these libraries using pip:

```
pip install scikit-learn numpy pandas
```

## Usage

Ensure that the relevant dataset is present in your working directory. Run the Python script, which will load the data, preprocess it if necessary, train the Ridge Regression model using K-Fold Cross-Validation, and report the average RMSE for each lambda value.

Remember to modify the lambda hyperparameters and the number of folds in the K-Fold Cross-Validation as needed to fit the needs of your specific dataset.
