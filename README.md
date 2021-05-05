# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about customers of a bank and we seek to predict a binary variable (yes/no) which measure if the customer bought a financial product or not.

The best performing model was fitted by AutoML: VotingEnsemble with 0.9168 accuracy.

## Scikit-learn Pipeline
The pipeline architecture includes the following steps:
1. **Training script**: 
  - in the train.py script we import the raw data using TabularDatasetFactory functionality from the azureml library, 
  - clean data using teh provided clean_data python function (missing values), 
  - generate new features (feature engineering, also using the clean_data python function, one-hot encoding and process date variables),
  - split data in train and test sets and 
  - fit model using scikit-learn python library
2. **Sklearn estimator**: 
  - we use the logistic regression model from the scikit-learn python library.
4. **HyperDrive**: 
  - we perform random search to optimize the parameters C and max_iter, 
  - set accuracy as primary metric to compare fitted models, 
  - set bandit policy and 
  - set the sklearn estimator (logistic regression).
5. **Save Best Model**: the best logistic regression we got has params C=0.85 and max_iter=100, and the performance was accuracy=0.85.

The benefits of random search are the following: 
- We can improve the performance os our model by searching the best combination of hyper-parameters,
- support continous and dicrete distributions and
- deeper sarch of each param space than grid search

The benefit of using a early stopping policy is:
- (Bandit) It terminates runs where the primary metric is not within the specified slack factor compared to the best performing run, so we save time.

## AutoML
1. **Import data**:
  - In the udacity-project.ipynb notebook we import tha raw data using TabularDatasetFactory functionality from the azureml library.
2. **Cleaning of data**:
  - In the udacity-project.ipynb notebook we clean data using teh provided clean_data python function (missing values).
3. **Feature engineering**:
  - In the udacity-project.ipynb notebook we perform feature engineering, also using the clean_data python function (one-hot encoding and process date variables).
4. **Splitting of data**:
  - split data in train and test sets.
5. **Configuration of AutoML**
6. **Save Best Model**: the best model using AutoML is an ensemble model (bagging) of the models trained: LightGBM, XGBoostClassifier, XGBoostClassifier, XGBoostClassifier, LightGBM, XGBoostClassifier and RandomForest.

![image](https://user-images.githubusercontent.com/32734434/117157619-050c8f00-adbf-11eb-9651-3b448ec332f4.png)

## Pipeline comparison
The pipeline of both solutions is very similar (for more details just compare the above list of steps) the only difference is the training configs:
1. the first solution: we fix the model type (logistic regression) and leverage HyperDrive to fine-tune hyper-parameters,
2. the second solution: AutoML try different types of models (logistic regressions, XGBoost, Random Forest, Neural Nets, Ensembles...) using their optimal hyper-parameters.

## Future work
- Deep feature engineering combining variables in the dataset,
- Use External data of each customer,
- Deep model interpretability to completely understand the model behaviour.

## Proof of cluster clean up
![image](https://user-images.githubusercontent.com/32734434/117159195-608b4c80-adc0-11eb-9959-7dadd3bf13a9.png)
