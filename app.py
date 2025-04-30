from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input array from JSON body
        data = request.get_json()
        input_array = data.get('data', [])

        if len(input_array) != 29:
            raise ValueError(f"Expected 29 values, but got {len(input_array)}.")

        input_array = [float(x) for x in input_array]
        prediction = model.predict([input_array])[0]
        verdict = 'Fraudulent' if prediction == 1 else 'Legitimate'

        # Redirect to result page with verdict
        return {'redirect': url_for('result', verdict=verdict)}
    except Exception as e:
        return {'redirect': url_for('result', verdict=f"Error: {str(e)}")}

@app.route('/result')
def result():
    verdict = request.args.get('verdict', 'No result')
    return render_template('result.html', prediction_text=verdict)

if __name__ == '__main__':
    app.run(debug=True)
