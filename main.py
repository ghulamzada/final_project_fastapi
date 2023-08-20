from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import pandas as pd
import joblib
from model.data import process_data
from model.model import inference

app = FastAPI()

# Define a Pydantic model to represent the request body
class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    hours_per_week: int
    native_country: str
    salary: str

# Load the trained model
model = joblib.load("model/trained_model.joblib")

# Load the encoder and label binarizer
encoder = joblib.load("model/encoder.pkl")
label_binarizer = joblib.load("model/label_binarizer.pkl")

# Define the root endpoint for GET request
@app.get("/")
async def root():
    return {"message": "Welcome to the API for model inference."}

exmaple_in_swagger_api = {
  "age": 31,
  "workclass": "Local-gov",
  "fnlgt": 42346,
  "education": "Masters",
  "education_num": 14,
  "marital_status": "Married-civ-spouse",
  "occupation": "Prof-specialty",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "hours_per_week": 50,
  "native_country": "United-States",
  "salary": "<=50K"
}
# Define the endpoint for model inference using POST request
@app.post("/inference/")
async def model_inference(input_data: InputData= Body(default=exmaple_in_swagger_api)):
    try:
        # Convert the input data to a DataFrame
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])

        # Process the input data using the process_data function
        processed_data, _, _, _ = process_data(
            input_df,
            categorical_features=["workclass", "education", "marital_status",
                                  "occupation", "relationship", "race", "sex",
                                  "native_country"],
            label="salary",
            training=False,
            encoder=encoder,
            lb=label_binarizer
        )

        # Perform any necessary data transformations or validations
        # For example, ensure the column order matches the model's expectations
        #processed_data = processed_data[model.feature_names]

        # Perform model inference
        predictions = inference(model,processed_data)

        # Return the predictions
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)