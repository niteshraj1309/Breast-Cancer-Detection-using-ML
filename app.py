import pickle
import json
from flask import Flask, request, jsonify, send_file
import numpy as np

app = Flask(__name__)

@app.route('/')
def serve_ui():
    return send_file('templates/index.html')  # serves the UI page

model = 'breast_cancer_detector.pickle'

# Load trained model
with open(model, 'rb') as f:
    model = pickle.load(f)

# Load feature means
with open('feature_means.json', 'r') as f:
    feature_means = json.load(f)

# Full feature list (in correct order)
feature_order = list(feature_means.keys())

# Which features are coming from frontend
input_features = [
    'radius_mean', 'texture_mean', 'perimeter_mean',
    'area_mean', 'smoothness_mean'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_values = data.get('features', [])

    if len(input_values) != len(input_features):
        return jsonify({'error': 'Expected 5 input features'}), 400

    # Create feature dictionary
    input_dict = dict(zip(input_features, input_values))
    full_input = feature_means.copy()
    full_input.update(input_dict)  # Replace 5 with user inputs

    # Create final ordered feature vector
    features_vector = [full_input[f] for f in feature_order]

    pred = model.predict([features_vector])[0]
    result = 'Malignant' if pred == 'M' else 'Benign'

    return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(debug=True)