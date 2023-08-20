
from model import train_model, inference, calculate_data_slice_performance
import pandas as pd


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

if __name__ == "__main__":

    calculate_data_slice_performance(train_model,
                                     inference,
                                     data,
                                     slice_features = cat_features,
                                     output_file="model/slice_output.txt")




