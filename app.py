from flask import Flask, render_template, request
import pickle

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask('__name__')
X_scaler, Y_scaler = pickle.load(open('X_scaler.pkl', 'rb')),  pickle.load(open('y_scaler.pkl', 'rb'))
model = pickle.load(open('desease_model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('test.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data =  pd.DataFrame([{
            'age' : request.form.get('age'),
            'weight' : request.form.get('weight'),
            'height' : request.form.get('height'),
            'blood_presure' : request.form.get('blood_pressure'),
            }])
    print(data)
    x = X_scaler.transform(data.values)   
    pred = model.predict(x)
    prediction = Y_scaler.inverse_transform(pred)
    return render_template('test.html', text=f'Predicted Disease is {prediction[0]}')


app.run()
