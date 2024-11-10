Sunday 10th Nov, 2024
###############################################################################################
Project Documentation: AI-Powered Health Analysis
###############################################################################################

## Project Overview

This project is an AI-driven health diagnosis tool that predicts possible diseases based on input parameters such as age, weight, height, and blood pressure. 
The project includes machine learning models, a Flask web application, and a front-end interface where users can input their details to receive an AI-powered health analysis.

## Directory Structure

* app.py: The main Flask application file that serves the web interface and handles predictions.
* Project.ipynb: Jupyter Notebook containing the data preprocessing, model training, and evaluation.
* general_disease_diagnosis.csv: Dataset used for training and testing the model.
* Final_submission.csv: Final dataset containing both training and testing data with predictions.
* cat_imputer.pkl, imputer.pkl, num_imputer.pkl: Serialized files for imputing missing values.
* disease_model.pkl: The trained machine learning model for disease prediction.
* X_scaler.pkl, y_scaler.pkl: Scaler files used for standardizing the feature data and encoding labels.
* templates/test.html: Front-end HTML file for the user interface.

## Workflow

* Data Loading: Load the dataset general_disease_diagnosis.csv using Pandas.
* Data Preprocessing:
- Separate numerical and categorical columns.
- Use IterativeImputer to handle missing values in numerical data.
- Impute missing values in categorical columns using the mode.
	
* Feature Scaling and Encoding:
- Apply StandardScaler to numerical features.
- Encode target labels (disease classes) with LabelEncoder.

* Model Training:
- Split the data into training and validation sets.
- Train multiple models (Random Forest, Gradient Boosting, SVM) using GridSearchCV for hyperparameter tuning.
- Select the best model based on F1 score.
	
* Model Evaluation:
- Evaluate the best model on validation data.
- Generate a final prediction on the test set.
	
* Model Saving:
- Save the trained model, scaler, and other preprocessing artifacts as .pkl files.

* Flask Application:
- Serve the model through a Flask app.
- Accept user input, preprocess it, make predictions, and display results on the front end.

###############################################################################################
## About Module
###############################################################################################

## 1. Project.ipynb (Jupyter Code / Model Training)

* Libraries: Imports necessary libraries like numpy, pandas, sklearn, and pickle.
- Data Loading: Loads the dataset from general_disease_diagnosis.csv.

* Data Preprocessing:
- Imputes missing values in numerical columns with IterativeImputer.
- Fills missing values in categorical columns with the mode.

* Feature Scaling and Encoding:
- Scales numerical features using StandardScaler.
- Encodes disease labels with LabelEncoder.

* Model Training:
- Trains multiple models (Random Forest, Gradient Boosting, and SVM) with hyperparameter tuning.
- Uses GridSearchCV to find the best parameters and selects the model with the highest F1 score.
- Model Evaluation:
- Evaluates the model on training and validation sets and prints accuracy and classification report.
- Model Saving: Saves the best model and preprocessing artifacts as .pkl files.

## app.py (Backend)

* Libraries: Imports Flask, pickle, pandas, and numpy.
- App Initialization: Initializes the Flask application.
* Endpoints:
- Renders the main page (test.html) for user input.
- predict: Accepts user input, preprocesses the data, makes predictions using the loaded model, and returns the result.
* Model Loading: Loads the saved scaler and model files (X_scaler.pkl, y_scaler.pkl, disease_model.pkl).

## test.html (UI)

* UI Components:
- HTML form to collect user inputs for age, weight, height, and blood pressure.
- Displays the prediction result after form submission.
- Styling: Links to external stylesheets for styling form components.

## Mian Files

=> disease_model.pkl: Serialized model for disease prediction.
=> X_scaler.pkl, y_scaler.pkl: Scalers used for standardizing input features and encoding labels.
=> imputer.pkl: Imputer used for filling missing values in the dataset.


###############################################################################################
How To Run
###############################################################################################

Step1: Open the project folder in your preferred IDE.
Step2: Set up the environment for Python 3.11 and open the terminal.
Step3: Run the command: python app.py.
Step4: In the terminal, you will see a localhost link. Open this link in your browser.
Step5: Use the HTML UI page to interact with the application.

###############################################################################################

Thank You
Team Black Hats

############################################################################################### 
