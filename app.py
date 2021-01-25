# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 10:02:54 2021

@author: ESFITA-USER
"""

# Importing Libraries:
import pandas as pd
from sklearn.externals import joblib 
import numpy as np
from flask import Flask, request,render_template
import warnings
import os
warnings.filterwarnings("ignore")
 
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Load the models from the directory 
mean_rf_from_joblib = joblib.load('mean_enc_rf_reg.pkl')
ordinal_rf_from_joblib = joblib.load('ordinal_enc_rf_reg.pkl')
mean_enc = joblib.load('mean_enc.pkl')
ordinal_enc = joblib.load('ordinal_enc.pkl')

@app.route('/',methods=['POST','GET'])
def home():
    return render_template('test.html')

@app.route('/back',methods=['POST','GET'])
def back():
    return render_template('test.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    predict_request = []
    data = []
    predict_request.append(request.form["state"])
    predict_request.append(request.form["brand"])
    predict_request.append(request.form["model"])
    predict_request.append(request.form["year"])
    
    data.append(request.form["state"])
    data.append(request.form["brand"])
    data.append(request.form["model"])
    data.append(request.form["year"])
    
    # Use the mean_enc model to make encoding:
    enc = mean_enc.transform(pd.DataFrame(data=[predict_request],columns=['State','Brand','Model','Year']))
    print(enc.values.tolist()[0])
    predict_request = enc.values.tolist()
    
    predict_request = np.array(predict_request)
    prediction = int(mean_rf_from_joblib.predict(predict_request)[0])
    
    return render_template('test_result.html', prediction_text='Vehicle Price Should be {} ₹'.format(prediction),data = data)
    

if __name__ == "__main__":
    app.run()