Cuisine Classification using Machine Learning
Project Overview

This project implements a machine learning model to classify restaurants based on their cuisine type.
The system preprocesses restaurant data, encodes categorical features, trains a Random Forest classifier, and evaluates the model using accuracy.
It also displays the top 10 cuisines with their names for analysis.

Objective

To build a machine learning model for restaurant cuisine classification

To preprocess real-world data by handling missing values

To encode categorical variables

To train and evaluate a classification model

To analyze the most common cuisines in the dataset

Dataset

File name: Dataset.csv

Contains restaurant-related attributes such as location, ratings, votes, cost, etc.

Target column: Cuisines

Technologies Used

Python 3

Pandas – Data manipulation

Scikit-learn – Machine learning models and evaluation

Methodology

Load the dataset

Handle missing values:

Numerical columns → Mean

Categorical columns → Mode

Encode categorical variables using Label Encoding

Split data into training and testing sets

Train a Random Forest Classifier

Evaluate the model using accuracy

Display the top 10 cuisines with names
