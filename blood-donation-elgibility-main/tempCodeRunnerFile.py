from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the pre-trained model directly
model = joblib.load('C:/Users/archi/Desktop/blood/blood-donation-elgibility-main/model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("Received JSON data:", data)

    # Extract features from JSON data
    features = [
        data['gender'],
        data['hemoglobinCount'],
        data['age'],
        data['weight'],
        data['underlyingDisease'],
        data['infections'],
        data['medications'],
        data['daysSinceAlcohol'],
        data['bloodDonationFrequency']
    ]

    # Make a prediction using the pre-trained model
    prediction = model.predict([features])
    
    #
# Render the result.html template with the prediction result
    return render_template('result.html', result=prediction[0])
    

if __name__ == '__main__':
    app.run(debug=True, port=3001)
