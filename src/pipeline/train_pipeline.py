import sys
import os
import pandas as pd

from datetime import datetime
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from src.components.data_ingestion import DataIngestion
from src.components.data_tranformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

@dataclass
class TrainPipelineConfig:
    train_data_path: str=os.path.join('newdata', 'train.csv')
    test_data_path: str=os.path.join('newdata', 'test.csv')
    raw_data_path:str=os.path.join('newdata', 'new_data.csv')
    preprocessor_obj_file_path: str=os.path.join('newdata','preprocessor.pkl')
    trained_model_file_path: str=os.path.join('newdata', 'model.pkl')
    #
    old_model: str=os.path.join('artifacts', 'model.pkl')
    


class TrainPipeline:
    def __init__(self) -> None:
        self.data_trainpipeline_config = TrainPipelineConfig()
        pass

    def get_new_train_test_data(self):

        getdata = DataIngestion()
        setattr(getdata, 'ingestion_config', TrainPipelineConfig())

        train_data, test_data = getdata.initiate_data_ingestion()
        return
    
    def transform_data(self):

        datatranf = DataTransformation()
        setattr(datatranf,'data_transformation_config',TrainPipelineConfig())

        train_arr, test_arr, _ = datatranf.initiate_data_transformation(
                                            train_path = self.data_trainpipeline_config.train_data_path,
                                            test_path = self.data_trainpipeline_config.test_data_path,
                                        )
        return {'train_arr':train_arr, 
                'test_arr':test_arr, 
                'preprocessor_path':_}


    def train_new_model(self, train_array, test_array):

        model_train = ModelTrainer()
        setattr(model_train, 'model_trainer_config', TrainPipelineConfig())

        model_report, models = model_train.initiate_model_trainer(
                                                    train_array=train_array, 
                                                    test_array=test_array
                                                    )
        return {'model_report': model_report, 'models':models}
    
    def get_model_metrics(self,model_report, models):

        model_train = ModelTrainer()
        setattr(model_train, 'model_trainer_config', TrainPipelineConfig())

        best_model_score = model_train.get_best_model(model_report=model_report,
                                   models=models,)
        return best_model_score
    
    def tunig_model(self):
        pass

    def evaluate_models(self, train_array, test_array):
        
        try:
            new_model = load_object(self.data_trainpipeline_config.trained_model_file_path)
            old_model = load_object(self.data_trainpipeline_config.old_model)

            X_train,y_train,X_test,y_test = (
                                            train_array[:,:-1],
                                            train_array[:,-1],
                                            test_array[:,:-1],
                                            test_array[:,-1],                
                                            )

            # Fit on the new data 
            old_model.fit(X_train, y_train)

            # Prediction
            #old model
            y_old_train_pred = old_model.predict(X_train)
            y_old_test_pred = old_model.predict(X_test)

            old_mse = mean_squared_error(y_test, y_old_test_pred)
            old_r2 = r2_score(y_test, y_old_test_pred)
            
            #new model
            y_new_pred_train = new_model.predict(X_train)
            y_new_pred_test = new_model.predict(X_test)

            new_mse = mean_squared_error(y_test, y_new_pred_test)
            new_r2 = r2_score(y_test, y_new_pred_test)

            if new_mse < old_mse:

                dt_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                # Move old model to archive
                save_object(file_path='model_archive/model_'+ dt_time +'.pkl',
                            obj = old_model)
                
                # Replace the old model for the New model
                save_object(
                    file_path=self.data_trainpipeline_config.old_model,
                    obj = new_model
                    )         
            return {'new_metric': new_mse, 'old_metric': old_mse}
            
        except Exception as e:
            raise CustomException(e, sys)        
    
    def main(self):

        pipe = TrainPipeline()

        step1 = pipe.get_new_train_test_data()

        #Prepare the data
        step2 = pipe.transform_data()

        #Train new model
        step3 = pipe.train_new_model(train_array=step2['train_arr'], 
                                    test_array=step2['test_arr'],
                                    )
        # Get the best model
        step4 = pipe.get_model_metrics(model_report=step3['model_report'],
                                        models=step3['models'],
                                        )
        
        # tuning the best model
        #step5 = pipe.tunig_model()
        
        # Evaluate the best model new or old one
        step6 = pipe.evaluate_models(train_array=step2['train_arr'], 
                                    test_array=step2['test_arr'])         
                   
        return step6

# if __name__   == '__main__':

#     pipe = TrainPipeline()

#     #Get and split data
#     step1 = pipe.get_new_train_test_data()

#     #Prepare the data
#     step2 = pipe.transform_data()

#     #Train new model
#     step3 = pipe.train_new_model(train_array=step2['train_arr'], 
#                                 test_array=step2['test_arr'],
#                                 )
#     # Get the best model
#     step4 = pipe.get_model_metrics(model_report=step3['model_report'],
#                                     models=step3['models'],
#                                     )
    
#     # tuning the best model
#     #step5 = pipe.tunig_model()
    
#     # Evaluate the best model new or old one
#     step6 = pipe.evaluate_models(train_array=step2['train_arr'], 
#                                 test_array=step2['test_arr'])

#     pass
    
