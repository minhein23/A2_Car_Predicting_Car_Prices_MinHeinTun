🚗 A2 - Predicting Car Prices

By Min Hein Tun - IM - AIT

Please see the car price prediction website at <<https://st125367.ml.brain.cs.ait.ac.th/>>. It runs on the CSIM MLflow server.
📌 When you see the message "The connection is not private", you can click show details on your browser and go with unsafe. I will make sure to solve this problem in A3. 
 
📌 Project Overview
This project predicts car prices based on vehicle attributes such as engine size, mileage, and model.
It includes two models:

A1 Model: Original model from A1 assignment.
A2 Model: Improved model with optimized hyperparameters.
The Flask web application allows users to choose between both models for price prediction.

This project uses a machine learning model to predict car prices based on various features like engine size, max power, car age, and other car specifications. The model is built using Random Forest Regressor and is deployed through a Flask web application that allows users to input car features and receive predicted prices.


Project Overview
The goal of this project is to predict the selling price of a car based on features such as:

Brand (extracted from the car's name)
Engine size (in CC)
Mileage
The dataset is preprocessed to remove irrelevant or erroneous values, and a Linear Regression is used to predict the log-transformed selling price. The log transformation stabilizes the variance of the target variable.

After training the model, a Flask web app is developed to allow users to input car details and get a predicted price.

Dataset
The dataset used in this project contains information about cars, including:

Name: Car name (used to extract the brand)
Year: Year of manufacture
Selling Price: The target variable (price we want to predict)
KM Driven: The total kilometers driven by the car
Fuel: Fuel type (Diesel, Petrol, CNG, LPG)
Seller Type: Individual or Dealer
Transmission: Transmission type (Manual, Automatic)
Owner: Number of previous owners
Mileage: Car mileage (km per liter)
Engine: Engine capacity (in CC)
Max Power: Maximum power (in bhp)
Torque: Torque of the engine (not used in this model)
The data is preprocessed to remove CNG and LPG cars (due to a different mileage system) and non-relevant columns like Torque.

Features
The following features were selected to predict the car's selling price:

Year: The year the car was manufactured. It helps determine the car's age, which affects its value.
Engine: The engine size (in CC), which is a major factor in determining the car's price.
Max Power: The engine power (in bhp), which also correlates with the car’s value.
Brand: Extracted from the Name column, the car's brand can significantly impact its price.
The target variable is selling_price (log-transformed for stability).

Data Preprocessing
The following preprocessing steps were applied to the dataset:

Owner Mapping: The owner column was mapped to numerical values (e.g., "First Owner" → 1, "Second Owner" → 2).
Fuel Type Filtering: Rows with fuel types CNG or LPG were removed, as they use a different mileage system.
Cleaning Columns:
mileage, engine, and max_power were cleaned to remove non-numeric units (like "kmpl", "CC", and "bhp").
The brand was extracted from the Name column by keeping only the first word.
Log Transformation: The selling_price column was log-transformed to stabilize its variance.
Model Building
The machine learning model used in this project is a Random Forest Regressor. This algorithm was chosen because:

It can handle both linear and non-linear relationships.
It works well with a mix of numerical and categorical features.
It is robust to overfitting and performs well in most regression tasks.

Saving the Model: The trained model was saved using joblib to be used in the web application.

🚗 Car Price Prediction (A2) - Flask & Docker Deployment

⚙️ Tech Stack
Machine Learning: Scikit-Learn, NumPy, Pandas
Web Framework: Flask
Deployment: Docker
Model Tracking: MLflow

Web Application
A Flask web app was created to allow users to input car details (like year, engine size, and max power) and get the predicted car price. The web app includes:

A form to collect the input data.
A backend that loads the trained model and makes predictions based on the input features.
Result display that shows the predicted car price on the same page.


Technologies Used:
Flask: Web framework for the app.
joblib: For saving and loading the trained machine learning model.
HTML/CSS: For the frontend (form and results display).
How to Run
Install Dependencies: Make sure all dependencies are installed by running:


pip install pandas numpy scikit-learn flask joblib
Load the Dataset: Ensure the dataset (car_data.csv) is in the same directory as the script.

Run the Flask App: Run the following command to start the Flask app:
python app.py
Access the Web App: Open a web browser and go to http://127.0.0.1:5000/ to use the app.

Input Car Details: Enter the year, engine size, and max power in the form, and the app will display the predicted car price.

----Run with Docker----
To run the project using Docker, follow these steps:

Clone the repository (or copy the project folder).

Build the Docker image:

Make sure the app.py file and model file (random_forest_model.pkl) are in the same directory.
Create a Dockerfile with the following content:
    # Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]

Create a requirements.txt file with the following dependencies:
pandas
numpy
scikit-learn
flask
joblib

Build the Docker image:
docker build -t car-price-prediction .

Run the Docker container:
docker run -p 5000:5000 car-price-prediction

Access the Web App:
Once the Docker container is running, open your browser and go to:
http://127.0.0.1:5000/

📩 Contact
For issues or contributions, reach out at st125367@ait.asia