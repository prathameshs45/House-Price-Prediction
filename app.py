import flask
from flask import request, render_template
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model = pickle.load(open('myModel.pkl', 'rb'))
app = flask.Flask(__name__, template_folder='templates')
df = pd.read_csv('templates/pure_datset.csv')
locations = df['location'].unique().tolist()
@app.route('/')
def home():
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])
    sqft = int(request.form['sqft'])

    # Preprocess the input data (adjust as needed for your model)
    input_data = pd.DataFrame([[location, bhk, bath, sqft]], columns=['location', 'bhk', 'bath', 'total_sqft'])

    # Make the prediction
    prediction = model.predict(input_data)[0]*100000
   
    return render_template('index.html', prediction=prediction, locations=locations)


if __name__ == '__main__':
    app.run(debug=True)