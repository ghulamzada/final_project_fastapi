# Model Card

Project link on GitHub:
https://github.com/ghulamzada/final_project_fastapi.git

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
The value for fbeta is around 0.5, but it differs eachtime as the model is retrained.
\
\
**- precision**\
The value for precesion is slightly around 0.5 but it differs eachtime as the model is retrained. Ofcourse it can optimized and improved.
\
**- recall** 
The value for recal is also around 0.5 but it can also vary depending on each model retrain.

On data slices, however, the accuracy of scikit-learn package was used.
## Ethical Considerations
Usage of this model can have its ehtical aspects, however as this model is intended only for education purposes and not for commercial, therefore, the usage of this model for any education departments won't harm anyone and can be used without any problems. For commercial usages, this must be reviewd and in some cases needs to be optimzed - specially for data protection purposes.
## Caveats and Recommendations
This model can be improved using a ML-Model optimization i.e. Optuna to get the optimal accurcy, but as mentioned above, due to time-shortage I could not apply it just yet.