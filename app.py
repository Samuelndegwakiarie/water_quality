from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load dataset
df = pd.read_csv('/home/dynamic_dude/Genesis/water_quality/model/water_potability.csv')

# Assuming 'Potability' is your target variable
X = df.drop('Potability', axis=1)
y = df['Potability']

# Keep all nine features for fitting the scaler
X_subset = X.copy()

# Initialize StandardScaler
scaler = StandardScaler()

try:
    # Fit the scaler to the full dataset
    scaler.fit(X_subset)
except Exception as e:
    print(f"Error fitting the scaler: {e}")
    scaler = None

# Load the trained SVM model
model_path = 'model/svm_model.pkl'
try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model from {model_path}: {e}")
    model = None


@app.route('/')
def index():
    return render_template('index.html')


# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if scaler is None:
            return jsonify({'error': "Scaler is not fitted yet"}), 500

        features = []  # Initialize the features list
        for i in range(1, 10):  # Loop over all nine features
            feature = request.form.get(f'feature{i}')
            if feature:  # Check if the value is not empty
                features.append(float(feature))
            else:
                return jsonify({'error': "All nine features must be provided"}), 400

        scaled_features = scaler.transform([features])

        if model is None:
            return jsonify({'error': "Model is not loaded"}), 500

        # Print the input features for debugging
        print("Input features:", features)

        # Perform prediction using loaded model
        prediction = model.predict(scaled_features)

        # JSON response
        return jsonify({'prediction': prediction.tolist()})

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
