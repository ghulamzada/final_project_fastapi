# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf


## Model Details
Model developed by: Taj Mohammad Ghulam Zada
Model data: 09.07.2023
Model version: v.01
Model type: Decision Tree Classifier
Information about paramters: This is a basic model without hyperparameter optimization due to lack of time. Thus, only the default paramters of decision tree classifier was used

## Intended Use
To predict the  on the Census Income Data Set. 

## Training Data
Training Data Set Census Income Data and include 80% off all dataset available. 

## Evaluation Data
The evaulation data set is around 20% of all dataset
## Metrics
_Please include the metrics used and your model's performance on those metrics._
Following metrics were used:\
**- fbeta** \
**- precision** \
**- recall** 

On data slices, however, the accuracy of scikit-learn package was used.
## Ethical Considerations

## Caveats and Recommendations
This model can be improved using a ML-Model optimization i.e. Optuna to get the optimal accurcy, but as mentioned above, due to time-shortage I could not apply it just yet.