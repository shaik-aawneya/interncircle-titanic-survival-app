
# Titanic Survival Predictor

An interactive machine learning web application that estimates Titanic passenger survival probability using a Decision Tree Classifier.

## Project Overview

This application uses a machine learning model developed as part of Project 4 of the InternCircle Data Science & AI Internship.

Users can enter passenger details and receive a predicted survival result with an estimated probability.

## Features

- Interactive passenger input fields
- Passenger class selection
- Sex selection
- Age and fare sliders
- Real-time survival prediction
- Estimated survival probability
- Cached model loading

## Machine Learning Model

- Algorithm: Decision Tree Classifier
- Maximum depth: 3
- Dataset: Seaborn Titanic Dataset
- Features: pclass, sex, age, fare
- Target: survived

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## Model Performance

The selected model achieved 81.1% test accuracy in the Project 4 tree-depth comparison.

## Disclaimer

This application is an educational project based on historical Titanic passenger data. Its predictions are not factual or suitable for real-world decisions.
