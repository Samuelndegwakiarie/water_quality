from flask import Flask, request, jsonify, render_template
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
        # Handle form submission (if applicable)
        if request.method == 'POST':
            # Extract feature values from form data
            feature1 = request.form.get('feature1')
            feature2 = request.form.get('feature2')

            # Convert feature values to float (assuming numerical features)
            feature1 = float(feature1)
            feature2 = float(feature2)
        else:
            # Handle JSON data (if applicable)
            data = request.json
            # Extract feature values from JSON data
            feature1 = data.get('feature1')
            feature2 = data.get('feature2')

            # Convert feature values to float (assuming numerical features)
            feature1 = float(feature1)
            feature2 = float(feature2)

        # Perform prediction using the loaded model
        prediction = model.predict([[feature1, feature2]])

        # Return predictions as JSON response
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
