import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from math import ceil
from .helper import *


def get_metrics(data, target_variable, discrete_variables, private_data=None):
    pbar = tqdm(total=100)
    pbarCount = 0
    data_title = {"original":True, "synthetic": False, "private": False, "VM":False}
    
    analytical_utility = {"original" : 100}
    pbar.update(10)
    pbarCount += 10
    if target_variable in discrete_variables:
        regressor = False
    else:
        regressor = True
    pbar.update(10)
    pbarCount += 10
    X = list(data.columns)
    if "states_countries" in X:
            X.remove("states_countries")
    X.remove(target_variable)
    X = data[X]
    y = data[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
    mlModel, ml_score_O = ml_training(X_train, X_test, y_train, y_test, regressor=regressor)
    feat_importances_O = get_feature_importance(X, y, regressor=True)
    if "States" in feat_importances_O:
            feat_importances_O.drop("States", inplace=True)
    pbar.update(10)
    pbarCount += 10
    ml_utility = {"original" : ml_score_O, "regressor": regressor}
    feat_imp = {"original": feat_importances_O}
    # vulnerabilty_utility = {"original" : "45"}
    pbar.update(10)
    pbarCount += 10

    if private_data is not None:
        # analytical_utility["VM"]= analytical_utility_score(private_data)
        X = list(private_data.columns)
        if "states_countries" in X:
            X.remove("states_countries")
        X.remove(target_variable)
        X = private_data[X]
        y = private_data[target_variable]
        ml_score_V = ml_training(X, X_test, y, y_test, regressor=regressor)
        feat_importances_V = get_feature_importance(X, y, regressor=True)
        if "States" in feat_importances_V:
            feat_importances_V.drop("States", inplace=True)
        ml_utility["VM"] = ml_score_V # ceil((ml_score_V/ml_score_O)*100) 
        feat_imp["VM"]= feat_importances_V
        # vulnerabilty_utility["VM"] = 100 # vulnerabilty_metric(private_data)["Metric"]
        data_title["VM"] = True
        analytical_utility["gqi"] = general_quality_index(data, private_data)
    pbar.update(100-pbarCount)
    pbar.close()
    
    # return {"importantFeatures":feat_imp, "analytical_utility" : analytical_utility, "ml_utility" : ml_utility, "vulnerabilty_utility":vulnerabilty_utility, "data_title":data_title}
    return {"importantFeatures":feat_imp, "analytical_utility" : analytical_utility, "ml_utility" : ml_utility, "data_title":data_title}

