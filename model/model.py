from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from joblib import dump
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # Building Decision Tree model 
    trained_model_clf = DecisionTreeClassifier(random_state=42)
    return trained_model_clf.fit(X_train, y_train)



def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def save_trained_model(trained_model_clf, path_to_save: str):
    """
    Save trained model

    Input: trained model
    Output: saved trained ml model in "joblib" format
    """
    dump(trained_model_clf, path_to_save)

def calculate_data_slice_performance(train_model, inference, data, slice_features, output_file="slice_output.txt"):

    """
    Calculates the performance of a model on slices of categorical features.

    Parameters:
    - model: The trained model object that has a `predict` method.
    - data: The input data as a pandas DataFrame.
    - slice_features: A list of categorical features to slice on.

    Returns:
    - A dictionary mapping each categorical feature to its slice performance dictionary.
    """

    slice_performance = {}

    # Separate input features (X) and target variable (y)
    X = data.drop(slice_features, axis=1)
    y = data[slice_features]

    # Perform one-hot encoding on the input features
    ct = ColumnTransformer([('encoder', OneHotEncoder(), list(X.columns))], remainder='passthrough')
    X_encoded = ct.fit_transform(X)

    for feature in slice_features:
        slice_performance[feature] = {}

        # Get unique values of the current slice feature
        slice_values = data[feature].unique()

        for value in slice_values:
            # Create a mask for the current slice value
            mask = data[feature] == value

            # Apply the mask to the data
            sliced_data = data[mask]

            # Separate input features (X_slice) and target variable (y_slice)
            X_slice = sliced_data.drop(slice_features, axis=1)
            y_slice = sliced_data[feature]

            # Perform one-hot encoding on the sliced input features
            X_slice_encoded = ct.transform(X_slice)

            # Train the model
            model = train_model(X_encoded, y[feature])

            # Make predictions on the sliced data
            y_pred = inference(model, X_slice_encoded)

            # Calculate accuracy score for the slice
            performance = accuracy_score(y_slice, y_pred)

            # Store the performance in the dictionary
            slice_performance[feature][value] = performance

    # Print the performance for each slice combination
    with open(output_file, 'w') as f:
        for feature, performance_dict in slice_performance.items():
            f.write(f"Performance for slices of '{feature}':\n")
            print(f"Performance for slices of '{feature}':")
            for value, performance in performance_dict.items():
                f.write(f" - Slice value '{value}': {performance}\n")
                print(f" - Slice value '{value}': {performance}")


