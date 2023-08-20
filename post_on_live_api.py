import requests

# Define the API endpoint URL
api_url = "https://final-project-fastapi.onrender.com/inference/"  # Replace with the actual URL

# Sample input data
input_data = {
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

# Make the POST request
response = requests.post(api_url, json=input_data)

# Check the response status code and content
if response.status_code == 200:
    data = response.json()
    predictions = data["predictions"]
    print("Predictions:", predictions)
else:
    print("Error:", response.status_code, response.text)
