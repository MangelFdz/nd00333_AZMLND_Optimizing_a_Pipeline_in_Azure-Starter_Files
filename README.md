# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about customers of a bank and we seek to predict a binary variable (yes/no) which measure if the customer bought a financial product or not.

The best performing model was fitted by AutoML: VotingEnsemble with 0.9168 accuracy.

## Scikit-learn Pipeline
The pipeline architetcure includes the following steps:
1. **Training script**: 
  - in the train.py script we import the raw data, 
  - clean data using teh provided clean_data python function, 
  - generate new features (fature engineering, also using the clean_data python function),
  - split data in train and test sets and 
  - fit model using scikit-learn python library
3. **Sklearn estimator**: use the logistic regression model from the scikit-learn python library.
4. **HyperDrive**: 
  - we perform random search to optimize the parameters C and max_iter, 
  - set accuracy as primary metric to compare fitted models, 
  - set bandit policy and 
  - set the sklearn estimator (logistic regression).
6. **Best Model*: the best logistic regression we got has params C=0.85 and max_iter=100, and the performance was accuracy=0.85.

The benefits of random search are the following: 
- We can improve the performance os our model by searching the best combination of hyper-parameters,
- support continous and dicrete distributions

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
