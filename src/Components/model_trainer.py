import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
  AdaBoostRegressor,
  GradientBoostingRegressor,
  RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
  trained_model_file_path = os.path.join("artifact","model.pkl")

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config=ModelTrainerConfig()
  def initiate_model_trainer(self,train_array,test_array):
    try:
      logging.info("spliting and training test input data")
      x_train,y_train,x_test,y_test=(
        train_array[:,:-1],
        train_array[:,-1],
        test_array[:,:-1],
        test_array[:,-1]
      )
      model = {
        "Random Forest": RandomForestRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
       "Linear Regressor": LinearRegression(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "XGB Regressor": XGBRegressor(),
        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
        "AdaBoost Regressor": AdaBoostRegressor(),
      }

      params = {

                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                },

                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },

                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                },

                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"]
                },

                "XGB Regressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0]
                },

                "CatBoosting Regressor": {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.03, 0.05, 0.1],
                    "iterations": [100, 200]
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1, 1.0]
                }
            }

            # -----------------------------
            # MODEL EVALUATION
            # -----------------------------

      model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,y_test=y_test,x_test=x_test,
                                       models=model,params=params)
      best_model_score = max(sorted(model_report.values()))
      best_model_name = list(model_report.keys())[
        list(model_report.values()).index(best_model_score)
      ]
      best_model = model[best_model_name]
      if best_model_score < 0.6:
        raise CustomException("No best model found")
      logging.info(f"Best model for training and testing data")    
      save_object(
        file_path=self.model_trainer_config.trained_model_file_path,
        obj=best_model
      )
      predicted=best_model.predict(x_test)
      score = r2_score(y_test,predicted)
      return score
    except CustomException as e:
      raise CustomException(e,sys)
  