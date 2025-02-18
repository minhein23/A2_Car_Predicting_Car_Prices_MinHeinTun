from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load both models and the scaler
# old_model = joblib.load("/Users/minheintun/MHT Projects/Machine Learning/A2 - Predicting Car Prices/A1_model.pkl")  # A1 Model
old_model= joblib.load("./A1_model.pkl",'rb')
new_model = joblib.load("./A2_Model.pkl",'rb')  # A2 Model
scaler = joblib.load("./scaler.pkl",'rb')  # Same scaler for both

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/new_model")
def new_model_page():
    return render_template("new_model.html")

@app.route("/old_model")
def old_model_page():
    return render_template("old_model.html")

@app.route('/predict', methods=['POST'])
def predict():
        try:
            # Get data from the form
            year = int(request.form['year'])
            engine = float(request.form['engine'])
            max_power = float(request.form['max_power'])

            # Prepare the features for prediction
            features = np.array([[year, engine, max_power]])

            # Predict the car price using the model
            prediction = model.predict(features)

            # Reverse the log transformation (if applied)
            predicted_price = np.exp(prediction[0])  # Reverse log if you log-transformed the target

            return render_template('index.html', predicted_price=f"Predicted Car Price: {predicted_price:,.2f}")
        
        except Exception as e:
            return render_template('index.html', error_message=f"Error: {str(e)}")

@app.route("/predict_new", methods=["POST"])
def predict_new():
    data = request.json["features"]
    data_scaled = scaler.transform([data])  # Scale input
    prediction = new_model.predict(data_scaled)
    return jsonify({"predicted_price": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=7000)
