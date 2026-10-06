import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer ##used to create pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
import os
@dataclass
class DataTransformationConfig:
  preprocessor_obj_filr_path=os.path.join('artifact',"preprocessor.pkl")

class DataTransformation:
  def __init__(self):
    self.data_transformation_config = DataTransformationConfig()

  def get_data_transformer_object(self):
    try:
      numerical_columns = ["writing score","reading score"]
      categorical_columns = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
      ]
      num_pipeline = Pipeline(
        steps=[
          ("imputer",SimpleImputer(strategy="median")),
          ("scaler",StandardScaler())
        ]
        
      )
      cat_pipeline=Pipeline(
        steps=[
          ("imputer",SimpleImputer(strategy="most_frequent")),
          ("one_hot_encoder",OneHotEncoder()),
          ("scaler",StandardScaler(with_mean=False))
        ]
      )
      logging.info(f"categorical columns:{categorical_columns}")
      logging.info(f"numerical columns : {numerical_columns}")
      preprocessor = ColumnTransformer(
       [
        ("num_pipeline", num_pipeline, numerical_columns),
        ("cat_pipeline", cat_pipeline, categorical_columns)
        ]
      )
      return preprocessor
    
    except Exception as e:
      raise CustomException(e,sys)

  def initiate_data_transformation(self,train_path,test_path):
    try:
      train_df = pd.read_csv(train_path)
      test_df = pd.read_csv(test_path)
      logging.info("Reading the test and test data completed")
      logging.info("Obtaining preprocessing object")
      preprocessor_obj = self.get_data_transformer_object()
      target_column_name ="math score"
      numerical_columns = ["writing score","reading score"]
      input_features_train_df = train_df.drop(columns=[target_column_name],axis=1)
      target_features_train_df = train_df[target_column_name]
      input_features_test_df = test_df.drop(columns=[target_column_name],axis=1)
      target_features_test_df = test_df[target_column_name]
      logging.info(
        f"Applying preprocessor object on training dataFrame and testing dataframe"
      )
      input_feature_train_arr = preprocessor_obj.fit_transform(input_features_train_df)
      input_feature_test_arr = preprocessor_obj.transform(input_features_test_df)
      train_arr = np.c_[
        input_feature_train_arr,np.array(target_features_train_df)
      ]
      test_arr = np.c_[input_feature_test_arr,np.array(target_features_test_df)]
      logging.info(f"saved preprocessing object")
      save_object(
        file_path = self.data_transformation_config.preprocessor_obj_filr_path,
        obj=preprocessor_obj
      )
      return(
        train_arr,
        test_arr,
        self.data_transformation_config.preprocessor_obj_filr_path
      )
    except Exception as e:
      raise CustomException(e,sys)