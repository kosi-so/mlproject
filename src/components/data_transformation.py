import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
# from src.components.data_ingestion import DataIngestion
from src.utils import save_object

@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function transforms the data
        """
        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
                                   'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder())
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

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_object=self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ['reading_score', 'writing_score']

            X_train = train_df.drop(columns=[target_column_name], axis =1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name], axis =1)
            y_test = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on train and test dataframe")
            
            X_train_array = preprocessing_object.fit_transform(X_train)
            X_test_array = preprocessing_object.transform(X_test)

            train_transformed = np.c_[X_train_array, np.array(y_train)]
            test_transformed = np.c_[X_test_array, np.array(y_test)]

            logging.info("Saved preprocessing Object")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_object)

            return(train_transformed, test_transformed,
                   self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)


# data_inj = DataIngestion()
# TRAIN_PATH, TEST_PATH = data_inj.initiate_data_ingestion()

# train_df = pd.read_csv(TRAIN_PATH)
# test_df = pd.read_csv(TEST_PATH)
