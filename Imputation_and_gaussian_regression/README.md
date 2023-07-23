# README.md

# Project Overview
This project implements a Gaussian Process Regression (GPR) model to predict an output variable 'price_CHF' from given training and test datasets. The project uses various preprocessing techniques such as standard scaling, one-hot encoding, and imputation to clean and transform the data into a suitable format for the model. After training, the model predictions are exported to a .csv file.

# Dependencies
To run this project, you'll need to have the following Python libraries installed:
- numpy
- pandas
- scikit-learn

# Usage
You can run this project by simply executing the Python script. You'll need to have two .csv files, 'train.csv' and 'test.csv', in your working directory. The program will read these files, clean and preprocess the data, train the model, and make predictions. The predictions are then saved to 'results.csv' in your working directory.

The main functions of this script are as follows:

- `data_loading_enc()`: This function reads the data from the .csv files, performs preprocessing and returns the training and test data. 
- `average_LR_RMSE()`: This function takes the training data, an array of kernels, the number of folds for k-fold cross-validation, and alpha value. It trains the GPR model using each kernel and calculates the average RMSE of the model for each kernel.
- `modeling_and_prediction()`: This function takes the training data, fits a GPR model and returns predictions for the test data.

The script is structured so that it loads the data, trains the model, and writes the predictions to a .csv file when run.

# Note
Ensure that 'train.csv' and 'test.csv' are present in your working directory before running the script. If you want to understand the structure of these files, print the first five rows of these dataframes using `print(train_df.head(5))` and `print(test_df.head(5))`.
