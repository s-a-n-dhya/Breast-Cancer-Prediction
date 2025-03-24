from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load trained model
model = pickle.load(open('breast_cancer_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Breast Cancer Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.json  # Expecting JSON input
        features = np.array(data["features"]).reshape(1, -1)  # Convert to 2D array

        # Get prediction probabilities
        probabilities = model.predict_proba(features)[0]  # Returns [prob_Benign, prob_Malignant]

        # Create response JSON
        response = {
            "prediction": "Benign" if probabilities[0] > probabilities[1] else "Malignant",
            "probability_benign": float(probabilities[0]),
            "probability_malignant": float(probabilities[1])
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
