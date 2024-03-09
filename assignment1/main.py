from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the saved model and scaler
model = joblib.load('iris_model.pkl')
scaler = joblib.load('iris_scaler.pkl')

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
async def predict_species(iris_data: IrisInput):
    # Convert the input data into a numpy array
    data_array = np.array([[iris_data.sepal_length, iris_data.sepal_width, iris_data.petal_length, iris_data.petal_width]])

    # Scale the input data using the loaded scaler
    scaled_data = scaler.transform(data_array)

    # Make predictions using the loaded model
    prediction = model.predict(scaled_data)

    # Map the predicted class label to the species name
    species_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    species_name = species_names[prediction[0]]

    return {"species_name": species_name}
