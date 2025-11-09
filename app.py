from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load your trained model
model = joblib.load('AI/donation_predictor.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features from incoming JSON
    recency = data.get('recency')
    frequency = data.get('frequency')
    monetary = data.get('monetary')
    time = data.get('time')

    # Format for prediction
    features = np.array([[recency, frequency, monetary, time]])
    prediction = model.predict(features)[0]

    return jsonify({'donated': bool(prediction)})

if __name__ == '__main__':
    app.run(debug=True)