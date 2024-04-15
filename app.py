from flask import Flask, flash, request, redirect, url_for, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
from hmmlearn import hmm

# Loading Models
hmm_model = joblib.load("hmm_model.joblib")

# Configuring Flask
app = Flask(__name__)
@app.route('/')
def home():
    homeimgpath='bg.jpg'
    homeimg = url_for('static', filename=homeimgpath)
    bgimgpath='bg.avif'
    bgimg = url_for('static', filename=bgimgpath)
    return render_template('pdanalysis.html', homeimg=homeimg, bgimg=bgimg)


########################### Result Functions ########################################


@app.route('/result', methods=['POST'])
def resultbc():
    if request.method == 'POST':
        Age = int(request.form['Age'])
        Height = float(request.form['Height'])
        Weight = float(request.form['Weight'])
        HoehnYahr = float(request.form['HoehnYahr'])
        UPDRS = float(request.form['UPDRS'])
        UPDRSM = float(request.form['UPDRSM'])
        TUAG = float(request.form['TUAG'])
        Speed_01 = float(request.form['Speed_01'])
        Speed_10 = float(request.form['Speed_10'])
        model = joblib.load('hmm_model.joblib')
        new_input = np.array([[Age,Height,Weight,HoehnYahr,UPDRS,UPDRSM,TUAG, Speed_01, Speed_10]])
        predicted_label = model.predict(new_input)
        if(predicted_label[0]==0):
            print("Predicted Label: Control Object")
        else:
            print("Predicted Label: Parkinson Disease")
        homeimgpath='bg.jpg'
        homeimg = url_for('static', filename=homeimgpath)
        bgimgpath='bg.avif'
        bgimg = url_for('static', filename=bgimgpath)
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Breast Cancer test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('result.html', homeimg=homeimg, bgimg=bgimg, Age=Age,Height=Height,Weight=Weight,HoehnYahr=HoehnYahr,UPDRS=UPDRS,UPDRSM=UPDRSM,TUAG=TUAG, Speed_01=Speed_01, Speed_10=Speed_10, r=predicted_label[0])

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)
