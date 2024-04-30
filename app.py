from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model
model = joblib.load('model/model.pkl')

# Load dataset
df = pd.read_csv('/home/dynamic_dude/Genesis/water_quality/model/water_potability.csv')

# Assuming 'Potability' is your target variable
X = df.drop('Potability', axis=1)
y = df['Potability']

# Initialize StandardScaler
scaler = StandardScaler()
scaler.fit(X)

@app.route('/')
def index():
    return render_template('index.html')  

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:       
        features = []  # Initialize the features list
        for i in range(1, 10):
            feature = request.form.get(f'feature{i}')
            if feature is not None:  # Check if the value is not None
                features.append(float(feature))
            else:
                # Handle the case where the value is None (e.g., provide a default value or skip)
                pass
            
        scaled_features = scaler.transform([features])

        # Perform prediction using loaded model
        prediction = model.predict(scaled_features)

        # JSON response
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
