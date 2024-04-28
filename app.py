from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('/home/dynamic_dude/Genesis/water_quality/model.pkl')

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML template

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({'error': 'Request must be in JSON format'}), 400

        # Get input data from request
        data = request.json

        # Check for required keys in the data
        if 'features' not in data:
            return jsonify({'error': 'Missing required key: features'}), 400

        # Preprocess input data if necessary
        # (This step will depend on how your model was trained)

        # Extract features from the data
        features = data['features']

        # Make predictions using the loaded model
        predictions = model.predict(features)

        # Return predictions as JSON response
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
