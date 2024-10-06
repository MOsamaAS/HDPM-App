import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

import os
print("Current Working Directory:", os.getcwd())

def clean_data(data):
    predictors = data.drop(["target"], axis = 1)
    diagnosis = data["target"]
    return predictors, diagnosis

def create_model(predictors, diagnosis):
    scaler = StandardScaler()
    predictors = scaler.fit_transform(predictors)
    predictors_train, predictors_test, diagnosis_train, diagnosis_test = train_test_split(
        predictors,
        diagnosis,
        test_size = .2,
        random_state = 42,
    )
    model = XGBClassifier(objective = 'binary:logistic', random_state = 42)
    model.fit(predictors_train, diagnosis_train)

    diagnosis_prediction = model.predict(predictors_test)
    print("Accuracy of Diagnosis: ", accuracy_score(diagnosis_test, diagnosis_prediction))
    print("Classification Report: \n", classification_report(diagnosis_test, diagnosis_prediction))

    return model, scaler

def main():
    data = pd.read_csv("Data/heart.csv")
    print(data.head())
    print(data.isnull().sum())
    predictors, diagnosis = clean_data(data)

    model, scaler = create_model(predictors, diagnosis)

    model_binary(model, scaler)

def model_binary(model, scaler):
    try:
        with open('ModPkl/model.pkl', 'wb') as file:
            pickle.dump(model, file)
        with open('ModPkl/scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)
        print("Pickle files created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
