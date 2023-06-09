import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_tranformation import DataTransformation
from src.components.data_tranformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig


@dataclass 
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'data.csv')

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        "Read data from de Data Base"
        logging.info('Starting the data ingestion method')
        
        try:
            df = pd.read_csv('notebook\data\stud.csv') ##TODO Connection with database to get the data
            logging.info('import data from data source as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train test slpit initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)          
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion  and train test split completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":

    obj=DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    #Data tranformation test
    data_transformation = DataTransformation()
    train_array, test_array ,_ = data_transformation.initiate_data_transformation(train_path=train_data, test_path=test_data)

    #train model
    model_trainer = ModelTrainer()
    model_report, models = model_trainer.initiate_model_trainer(train_array=train_array, test_array=test_array)
    print(model_trainer.get_best_model(model_report=model_report, models=models))
    

