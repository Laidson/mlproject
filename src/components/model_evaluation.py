#TODO implement the grid search

import os
import sys
import pandas as pd
from dataclasses import dataclass 

from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.linear_model import LinearRegression

from src.exception import CustomException
from src.components.data_tranformation import DataTransformation
from src.logger import logging
from src.utils import save_object, load_object

@dataclass
class ModelEvaluationConfig:
    model_hiper_tuning_path  = os.path.join('model_tunnig', 'model_tuned.pkl')
    model_hiper_tuning_params_path  = os.path.join('model_tunnig', 'model_tuned_params.pkl')

class ModelEvaluation:

    def __init__(self) -> None:
        self.model_tuning_path = ModelEvaluationConfig()

    def grid_search_linear_regressor(self, best_model, train_array, test_array):
        #https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/          
                       
        try:
            if type(best_model) == type(LinearRegression()):
                model = best_model
                X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]                
            )
            else: 
                sys.exit()

            # define evaluation
            cv = RepeatedKFold(
                n_splits=3,
                n_repeats=3,
                random_state=1,
            )
            # 10 = 0.8641582200410254
            # 5 = 0.8664455330603423
            # 3 = 0.8680565274470798

            # define the search space
            space = dict()
            #space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
            space['fit_intercept'] = [True, False]
            #space['normalize'] = [True, False]

            # define the search
            search = GridSearchCV(
                estimator = model,
                param_grid = space,
                cv=cv,
                n_jobs=-1,
            )

            logging.info('Setup of GridSearch params')

        except Exception as e:
            CustomException(e, sys)
            
        try:
            # execute search
            result = search.fit(X_train,y_train)
 
            best_params = result.best_params_

            logging.info('GridSearch | Find the best params for LienarRegressor')


            save_object(
                file_path=self.model_tuning_path.model_hiper_tuning_path,
                obj=model
            )

            save_object(
                file_path=self.model_tuning_path.model_hiper_tuning_params_path,
                obj=best_params
            )

            # summarize result
            print('Best Score: %s' % result.best_score_)
            print('Best Hyperparameters: %s' % result.best_params_)

            return best_params

        except Exception as e:
            CustomException(e, sys)

    def evaluate_tuned_models(self, train_array, test_array, model, param):
    
        try:
            report = {} 

            X_train,y_train,X_test,y_test = (
                                            train_array[:,:-1],
                                            train_array[:,-1],
                                            test_array[:,:-1],
                                            test_array[:,-1]                
                                            )          

            model.set_params(**param)
            model.fit(X_train, y_train) #train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[str(model).split('(')[0]] = test_model_score
        
            return report
        
        except Exception as e:
            CustomException(e, sys)

if __name__ == '__main__':

    best_model = load_object('artifacts/model.pkl')

    #Data tranformation test
    data_transformation = DataTransformation()
    train_data = 'artifacts/train.csv'
    test_data = 'artifacts/test.csv'
    train_array, test_array ,_ = data_transformation.initiate_data_transformation(train_path=train_data, test_path=test_data)

    #Grid search
    mde = ModelEvaluation()
    best_param = mde.grid_search_linear_regressor(best_model, train_array, test_array)

    tuned_model = load_object('model_tunnig/model_tuned.pkl')
    
    # model evaluation on test dataset
    mde.evaluate_tuned_models(train_array, test_array, tuned_model, best_param)

    #TODO get bestmodel pass to production


