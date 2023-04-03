import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, get_features_names_by_type

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:

    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()
        self.categorical_columns = None
        self.numerical_columns = None
        self.target_column  = None # TODO get attention with the target column name 
    
    def get_data_tranformer_object(self):

        """This function takes in a pandas dataframe and returns a ColumnTransformer object. 
            The ColumnTransformer object is a preprocessor that is used to transform the dataframe into a format that can be used for machine learning. 

            The function first identifies the numerical and categorical columns in the dataframe. 
            It then creates two pipelines, one for numerical columns and one for categorical columns. 
            The numerical pipeline consists of an imputer and a scaler, while the categorical pipeline consists of an imputer, one-hot encoder, and a scaler. 
            The two pipelines are then combined into a ColumnTransformer object and returned. 

            Parameters:
            df (pandas.DataFrame): The dataframe to be preprocessed.

            Returns:
            preprocessor (sklearn.compose.ColumnTransformer): The preprocessor object.
        """
        try:
            #TODO this is valid when you have a data set with numerical and categorical features if not should adapt
            numerical_columns = self.numerical_columns.copy()
            numerical_columns.remove(self.target_column) # TODO undertand if you need to remove the target column on the right list
            categorical_columns =self.categorical_columns.copy()
            logging.info('get the numerical and cat column names')

        except Exception as e:
            raise CustomException(e, sys)

        try:
            num_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            self.categorical_columns = get_features_names_by_type(df=test_df, numerical=False)
            self.numerical_columns = get_features_names_by_type(df=test_df, numerical=True)
            self.target_column = 'math_score'

            logging.info('Read train test data completed')
            logging.info('Starting the preprocessing object')
            
            #deleting traget column
            input_feature_train_df = train_df.drop(columns=[self.target_column], axis=1)
            target_feature_train_df = train_df[self.target_column]

            input_feature_test_df = test_df.drop(columns=[self.target_column], axis=1)
            target_feature_test_df = test_df[self.target_column]

            logging.info(
                    f"Applying preprocessing object on training dataframe and testing dataframe."
                )
            
            preprocessing_object = self.get_data_tranformer_object()
            
            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df) 
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df )]

            logging.info('Save processing objects')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_object

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)            
        
      