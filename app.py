from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('placement_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    dsa_score = float(request.form['dsa_score'])
    resume_score = float(request.form['resume_score'])
    communication_score = float(request.form['communication_score'])
    development_score = float(request.form['development_score'])
    college_tier = float(request.form['college_tier'])
    
    # Prepare input features
    features = np.array([[dsa_score, resume_score, communication_score, development_score, college_tier]])
    
    # Scale features
    scaled_features = scaler.transform(features)
    
    # Predict using the model
    prediction = model.predict(scaled_features)
    
    # Interpret prediction
    result = 'Placed' if prediction[0] == 1 else 'Not Placed'
    
    return render_template('result.html', prediction_text=f'Placement prediction: {result}')

if __name__ == '__main__':
    app.run(debug=True)
