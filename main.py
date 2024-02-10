# Import necessary libraries
import pandas as pd
from flask import Flask, render_template, request
import pickle

# Initialize Flask application
app = Flask(__name__)
data=pd.read_csv('data.csv')

# Load the pre-trained machine learning model from a pickled file
with open("housing_price_prediction_using_dynamic_pricing.pkl", 'rb') as file:
    pipe = pickle.load(file)

# Define the default route that renders the index.html template
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for making predictions based on user input
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve user input from the HTML form
    bedrooms = request.form.get('bhk')
    bathrooms = request.form.get('bath')
    sqft_living = request.form.get('sqftliv')
    sqft_lot = request.form.get('sqftlot')
    floors = request.form.get('floors')
    waterfront = request.form.get('waterfront')
    view = request.form.get('view')
    condition = request.form.get('condition')

    # Create a DataFrame from the input data
    input_data = pd.DataFrame([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition]],
                               columns=['bhk', 'bath', 'sqftliv', 'sqftlot', 'floors', 'waterfront', 'view', 'condition'])
    
    # Convert input data to float
    input_data = input_data.astype(float)

    # Make predictions using the loaded model (pipe)
    prediction = pipe.predict(input_data)[0]
    
    # Return the prediction as a string
    return str(prediction)

# Run the Flask application if the script is executed
if __name__ == '__main__':
    # Start the application in debug mode on port 5001
    app.run(debug=True, port=5001)
