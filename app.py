from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler,OneHotEncoder

import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open('sal_pred.pkl', 'rb'))
pre_processor_model = pickle.load(open('preprocessor.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Age = float(request.form['AGE'])
        Leaves_used = float(request.form['LEAVES USED'])
        Ratings = float(request.form['RATINGS'])
        Past_Exp = float(request.form['PAST EXP'])
        Years_In_Office=float(request.form['YEARS IN OFFICE'])
        SEX = request.form['SEX']
        DESIGNATION = request.form['DESIGNATION']
        UNIT = request.form['UNIT']

        data = {
            'SEX': SEX,
            'DESIGNATION': DESIGNATION,
            'AGE': Age,
            'UNIT': UNIT,
            'LEAVES USED': Leaves_used,
            'RATINGS': Ratings,
            'PAST EXP': Past_Exp,
            'YEARS IN OFFICE': Years_In_Office,
        }

        print(data)

        df = pd.DataFrame([data])
        print(df)
        features=pre_processor_model.transform(df)
        print(features)
        prediction = model.predict(features)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text="The Predicted Salary is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)