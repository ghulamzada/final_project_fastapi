# Script to train machine learning model.
from model.data import process_data
from sklearn.model_selection import train_test_split
from model.model import train_model, compute_model_metrics, inference

# Add the necessary imports for the starter code.
import pandas as pd
from joblib import dump


# Add code to load in the data.
data = pd.read_csv("data/census.csv", skipinitialspace = True)

# Droping duplicate rows
data.drop_duplicates(inplace=True)

# Removing 2 unneeded columns
data.drop(['capital-gain', 'capital-loss'], axis=1, inplace=True)

data.rename(columns={
    "education-num": "education_num",
    "marital-status": "marital_status",
    "hours-per-week": "hours_per_week",
    "native-country": "native_country"
}, inplace=True)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

# Proces the test data with the process_data function.
cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.
trained_model = train_model(X_train, y_train)

# Creating X_test and y_test for model evaluation
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

# Creating prediction
y_pred = inference(trained_model, X_test)

# Creating model metrics (beta, precision, recall)
fbeta, precision, recall = compute_model_metrics(y_test, y_pred)

# Save the model to a file
dump(trained_model, 'model/trained_model.joblib')




