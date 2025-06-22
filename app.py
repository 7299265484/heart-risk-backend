from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("heart_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array([[data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'],
                          data['fbs'], data['restecg'], data['thalach'], data['exang'],
                          data['oldpeak'], data['slope'], data['ca'], data['thal']]])
    prediction = model.predict(features)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

