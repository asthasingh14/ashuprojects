# from flask import Flask, request, jsonify
from flask import Flask, request, jsonify

from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)
# Load the pre-trained model directly
model = joblib.load("C:/stark/blood donation/trained_ml_model.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("Received JSON data:", data)

    # Extract features from JSON data
    features = [
        
        data['hemoglobinCount'],
        data['age'],
        data['weight'],
        data['daysSinceAlcohol'],
        data['bloodDonationFrequency'],
        data['gender'],
        data['underlyingDisease'],
        data['infections'],
        data['medications'],
        
        
    ]

    # Make a prediction using the pre-trained model
    prediction = model.predict([features])
    
    print("The prediction is:",prediction)
    return jsonify({'prediction': prediction.item()})
# Render the result.html template with the prediction result
    # return render_template('result.html', result=prediction[0])
    

if __name__ == '__main__':
    app.run(debug=True, port=3001)



# from flask import Flask, request, jsonify, render_template, redirect, url_for
# from flask_cors import CORS
# import joblib

# app = Flask(__name__)
# CORS(app)

# # Load the pre-trained model directly
# model = joblib.load("C:/stark/blood donation/trained_ml_model.joblib")

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     print("Received JSON data:", data)

#     # Extract features from JSON data
#     features = [
#         data['hemoglobinCount'],
#         data['age'],
#         data['weight'],
#         data['daysSinceAlcohol'],
#         data['bloodDonationFrequency'],
#         data['gender'],
#         data['underlyingDisease'],
#         data['infections'],
#         data['medications'],
#     ]

#     # Make a prediction using the pre-trained model
#     prediction = model.predict([features])
    
#     print("The prediction is:", prediction)
    
#     # Return the prediction as a JSON response
#     return jsonify({'prediction': prediction[0]})

# @app.route('/result')
# def result():
#     prediction = request.args.get('prediction')
#     return render_template('result.html', result=prediction)

# if __name__ == '__main__':
#     app.run(debug=True, port=3001)
