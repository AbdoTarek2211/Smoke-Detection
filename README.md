Smoke Detection System

This repository contains code for a smoke detection system implemented in Python. The system utilizes machine learning algorithms to detect the presence of smoke based on sensor data.
Overview

The smoke detection system is built using Python and various libraries such as Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and Plotly. It employs the following machine learning models:

    Logistic Regression: A classification algorithm used to predict the presence of smoke based on sensor readings.
    Decision Tree: Another classification algorithm that creates a tree-like structure to make decisions based on input features.
    K-Nearest Neighbors (KNN): A non-parametric classification algorithm that classifies new data points based on the majority class of their k-nearest neighbors in the feature space.

Dataset

The system uses a dataset (smoke_dataset.csv) containing sensor readings such as temperature, humidity, TVOC (Total Volatile Organic Compounds), eCO2 (Equivalent Carbon Dioxide), and others. The dataset also includes a binary target variable indicating the presence of smoke (Fire Alarm).
Contents

    smoke_detection.ipynb: Jupyter Notebook containing the Python code for data preprocessing, exploratory data analysis, model training, evaluation, and prediction.
    logistic_regression_file: Serialized file containing the trained Logistic Regression model.
    Readme.md: This file, providing an overview of the project and its contents.

How to Use

    Clone the repository to your local machine.
    Ensure you have Python installed along with the required libraries listed in requirements.txt.
    Open and run the smoke_detection.ipynb notebook using Jupyter Notebook or any compatible environment.
    Follow the instructions in the notebook to preprocess the data, train the machine learning models, and evaluate their performance.
    Use the trained models to predict the presence of smoke based on new sensor data.

Requirements

The project requires Python 3.x along with the following libraries:

    Pandas
    NumPy
    Matplotlib
    Seaborn
    Scikit-learn
    Plotly
    Joblib

You can install the required libraries using pip:

pip install -r requirements.txt

Credits

This project was created by [Your Name].
