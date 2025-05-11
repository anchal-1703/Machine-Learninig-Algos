#!/usr/bin/env python
# coding: utf-8

# ## Real estate price predictor

#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


housing = pd.read_csv("data/data.csv")

# housing.hist(bins=50 , figsize =(20,15))


# train-Test splitting 


from sklearn.model_selection import train_test_split

train_set,test_set= train_test_split(housing,test_size=0.2,random_state=42)



# shuffeled split so that during training set evry possiblity covers

from sklearn.model_selection import StratifiedShuffleSplit

shuff = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in shuff.split(housing,housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


housing = strat_train_set.copy()
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge

models ={
 "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Decision Tree Regressor" : DecisionTreeRegressor(),
    "SVR" : SVR(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "Bayesian Ridge": BayesianRidge()

}
Model = next(iter(models))
Scores = 0
from sklearn.model_selection import cross_val_score

X = housing.drop('MEDV',axis=1)
y = housing['MEDV']
with open("regression_results.txt", "w") as file:
    for name,model in models.items():
        model.fit(X,y)
        predictions = model.predict(X)
        scores = cross_val_score(model,X,y,scoring='neg_mean_squared_error',cv=10)
        rmse_scores = np.sqrt(-scores)

        if(rmse_scores.mean() < Scores):
            Scores = rmse_scores
            Model = model
        
        output = f"Model: {name}\n"
        output +=f"Scores: {rmse_scores}\n"
        output +=f"Mean: {rmse_scores.mean()}\n"
        output +=f"Standard deviation: {rmse_scores.std()}\n"
        output += "-"*40 + "\n"
        # Write to file

        file.write(output)
print("All model outputs have been written to regression_results.txt.")

print(f"The Best Fit model from the given proplem is {Model} with minimum mean: {Scores}")







