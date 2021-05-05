# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about customers of a bank and we seek to predict a binary variable (yes/no).

The best performing model was fitted by AutoML: VotingEnsemble with 0.9168 accuracy.

## Scikit-learn Pipeline
1. Training script: import data, clean data, split data and fit model.
2. Sklearn estimator: logistic regression.
3. HyperDrive: random search, accuracy as primary metric, bandit policy and sklearn estimator.
4. Best Model: logistic regression with params C=0.85 and max_iter=100, and accuracy=0.85.

The benefits of random search are the following: supporting continous and dicrete distributions and deeper search over all params.

## AutoML
1. Import data
2. Cleaning of data
3. Splitting of data
4. Configuration of AutoML

## Pipeline comparison
The best model using AutoML is an ensemble model (bagging) of the models trained: LightGBM, XGBoostClassifier, XGBoostClassifier, XGBoostClassifier, LightGBM, XGBoostClassifier, RandomForest. The pipeline of both solutions is very similar, the only difference is the training configs:
1. the first solution: logistic regression + hyperdrive
2. the second solution: AutoML

## Future work
- Deep feature engineering
- Use External data
- Deep model interpretability
