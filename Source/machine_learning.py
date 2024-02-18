from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

# hyperparameter tuning
from hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt.pyll import scope
import numpy as np
import neptune
from neptune.integrations.xgboost import NeptuneCallback

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import time




#Defining a function named 'train_model' that takes 'params' as input.
def train_model_xgboost(params, neptune_callback, train_x, train_y, test_x, test_y):
    start_time = time.time()

    model = xgb.XGBClassifier(params=params, num_boost_round=5000, verbose_eval=False, callbacks=[neptune_callback])

    run_time = time.time() - start_time

    model.fit(train_x, train_y)

    predictions_test = model.predict(test_x)
    mae = mean_absolute_error(test_y, predictions_test)

    return {'status': STATUS_OK, 'loss': mae}



# This function will perfrom grid search on randon forest model and return the best params of the model

def random_forest_classifier_grid_search(param_grid, x_train, y_train):
    

    # Create a random forest classifier
    rf = RandomForestClassifier()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(rf, param_grid, cv=2, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    # Print the best parameters found
    return grid_search.best_params_


  







  