import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = 'models/iris_logistic_regression_model.joblib'
model = None


def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Файл не найден: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)


@app.route('/')
def home():
    return "MLOps Iris Prediction API."


@app.route('/predict', methods=['POST'])
def predict():
    try:
        load_model()  

        input_data = request.get_json()
        expected_features = [
            'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
        ]

        for feature in expected_features:
            if feature not in input_data:
                return jsonify({
                    "error": f"Отсутствует ожидаемая функция: {feature}"
                }), 400

        input_df = pd.DataFrame([{
            feature: input_data[feature] for feature in expected_features
        }])

        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0].tolist()

        iris_classes = ['setosa', 'versicolor', 'virginica']
        predicted_class = iris_classes[prediction]

        return jsonify({
            "prediction": predicted_class,
            "prediction_proba": prediction_proba
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({
            "error": f"Произошла непредвиденная ошибка: {str(e)}"
        }), 500
