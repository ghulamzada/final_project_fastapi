from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Test get method : 1
def test_api_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["message"] == "Welcome to the API for model inference."


# Test post method : 2

def test_inference_post_api():
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
    response = client.post("/inference/", json=input_data)
    assert response.status_code == 200


# Test post method : 3

def test_invalid_input():
    # Test invalid input data (missing a required field)
    invalid_input = {
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
        # "salary": "<=50K"  # Missing salary field
    }

    response = client.post("/inference/", json=invalid_input)
    assert response.status_code == 422  # Unprocessable Entity

# Test post method : 4

def test_prediction_below_50k():
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

    response = client.post("/inference/", json=input_data)
    assert response.status_code == 200
    assert "predictions" in response.json()

    predictions = response.json()["predictions"]
    assert len(predictions) == 1


# Test post method : 5

def test_prediction_above_50k():
    input_data = {
        "age": 40,
        "workclass": "Private",
        "fnlgt": 78666,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Exec-managerial",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "hours_per_week": 60,
        "native_country": "United-States",
        "salary": ">50K"
    }

    response = client.post("/inference/", json=input_data)
    assert response.status_code == 200
    assert "predictions" in response.json()

    predictions = response.json()["predictions"]
    assert len(predictions) == 1
