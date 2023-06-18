import pandas as pd
import pytest
#from ml.model import train_model, save_trained_model, compute_model_metrics, inference
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
#ml.data import process_data#from ml.model import train_model



@pytest.fixture
def data():
    df = pd.DataFrame(
        {
        "age":{"0":30,"1":23,"2":32,"3":24,"4":50,"5":46,"6":42,"7":32,"8":21,"9":26},
        "workclass":{"0":"Never-worked","1":"Private","2":"Private","3":"Private","4":"Private","5":"Private","6":"Self-emp-not-inc","7":"Private","8":"?","9":"Local-gov"},
        "fnlgt":{"0":176673,"1":56774,"2":61898,"3":34568,"4":175339,"5":58683,"6":175674,"7":196342,"8":278391,"9":103148},
        "education":{"0":"HS-grad","1":"HS-grad","2":"11th","3":"Assoc-voc","4":"HS-grad","5":"Bachelors","6":"HS-grad","7":"Doctorate","8":"Some-college","9":"HS-grad"},
        "education-num":{"0":9,"1":9,"2":7,"3":11,"4":9,"5":13,"6":9,"7":16,"8":10,"9":9},
        "marital-status":{"0":"Married-civ-spouse","1":"Never-married","2":"Divorced","3":"Never-married","4":"Married-civ-spouse","5":"Married-civ-spouse","6":"Married-civ-spouse","7":"Never-married","8":"Never-married","9":"Never-married"},
        "occupation":{"0":"?","1":"Craft-repair","2":"Other-service","3":"Transport-moving","4":"Handlers-cleaners","5":"Exec-managerial","6":"Sales","7":"Prof-specialty","8":"?","9":"Adm-clerical"},
        "relationship":{"0":"Wife","1":"Own-child","2":"Unmarried","3":"Not-in-family","4":"Husband","5":"Husband","6":"Husband","7":"Not-in-family","8":"Own-child","9":"Not-in-family"},
        "race":{"0":"Black","1":"White","2":"White","3":"White","4":"White","5":"White","6":"White","7":"White","8":"White","9":"White"},
        "sex":{"0":"Female","1":"Male","2":"Female","3":"Male","4":"Male","5":"Male","6":"Male","7":"Male","8":"Male","9":"Female"},
        "capital-gain":{"0":0,"1":7688,"2":0,"3":0,"4":0,"5":0,"6":0,"7":0,"8":0,"9":7688},
        "capital-loss":{"0":0,"1":0,"2":0,"3":0,"4":0,"5":0,"6":0,"7":0,"8":0,"9":0},
        "hours-per-week":{"0":40,"1":40,"2":15,"3":40,"4":40,"5":55,"6":60,"7":40,"8":16,"9":40},
        "native-country":{"0":"United-States","1":"United-States","2":"United-States","3":"United-States","4":"United-States","5":"United-States","6":"United-States","7":"United-States","8":"United-States","9":"United-States"},
        "salary":{"0":"<=50K","1":"<=50K","2":"<=50K","3":"<=50K","4":">50K","5":">50K","6":"<=50K","7":">50K","8":"<=50K","9":"<=50K"}
        }
    )
    return df


def test_train_model(data):
    # Droping duplicate rows
    data.drop_duplicates(inplace=True)
    # Removing 2 unneeded columns
    data.drop(['capital-gain', 'capital-loss'], axis=1, inplace=True)
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)
    assert train.shape[1] == 13
    assert test.shape[1] == 13

    # Proces the test data with the process_data function.
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    assert X_train.shape == 31
    assert lb == 'binary'