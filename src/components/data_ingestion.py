import os  ## For creating a directory, to save the data in the local directory, in the form of a zip file
import sys ## Runtime
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass ## For creating a class and its attributes in a single line of code, to store the data
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',"train.csv") ## The data will be stored in the artifacts folder in the form of a train.csv file
    test_data_path: str = os.path.join('artifacts',"test.csv") ## The data will be stored in the artifacts folder in the form of a test.csv file
    raw_data_path: str = os.path.join('artifacts',"data.csv") ## The data will be stored in the artifacts folder in the form of a data.csv file

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        ## Data Ingestion Code used to read the data from the source and store it in the artifacts folder
        ## From various sources like github, s3, azure blob storage etc. 
        logging.info("Data ingestion started")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exeption as e:
            raise CustomException(e,sys)

if __name__ == "__main__":

    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array,test_array,_ = data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array,test_array))

