from flask import Flask
import joblib
import sklearn
import pandas as pd

app = Flask(__name__)

@app.route("/")
def test():
    loaded_model = joblib.load('../machine_learning_model/model.pkl')

    data = pd.read_csv("../dataset/student-mat.csv", sep=";")
    data = pd.get_dummies(data, drop_first=True)
    data = data.drop(columns='G3')

    data = data.iloc[0]

    try:
        pred = loaded_model.predict([data])
        return f"Prediction: {pred}"
    except ValueError as e:
        return f"Prediction error: {e}", 500