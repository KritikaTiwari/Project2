import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle


app = Flask(__name__)
model = pickle.load(open('house.pkl','rb')) 


@app.route('/')
def home():
  
    return render_template("show.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
  
    exp = float(request.args.get('exp'))
    exp1 = float(request.args.get('exp1'))
    exp2= float(request.args.get('exp2'))
    exp3= float(request.args.get('exp3'))
    exp4=float(request.args.get('exp4'))
    exp5=float(request.args.get('exp5'))
    prediction = model.predict([[exp,exp1,exp2,exp3,exp4,exp5]])

    return render_template('show.html', prediction_text='Regression Model  has predicted price for the house : {}'.format(prediction))
  
if __name__ == "__main__":
    app.run(debug=True)
