import os
import sys
import dill  

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):

    try:

        dir_path = os.path.dirname(file_path)

        os.makedirs(
            dir_path,
            exist_ok=True
        )

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:

        raise CustomException(e, sys)


def evaluate_model(
    x_train,
    y_train,
    x_test,
    y_test,
    models,
    params
):

    try:

        report = {}

        for model_name, model in models.items():

            # Get parameters for current model
            para = params.get(
                model_name,
                {}
            )

            # -----------------------------
            # HYPERPARAMETER TUNING
            # -----------------------------

            if para:

                gs = GridSearchCV(
                    estimator=model,
                    param_grid=para,
                    cv=3,
                    scoring="r2",
                    n_jobs=-1
                )

                gs.fit(
                    x_train,
                    y_train
                )

                # Set best parameters
                model.set_params(
                    **gs.best_params_
                )

                print(
                    f"\n{model_name}"
                )

                print(
                    f"Best Parameters: "
                    f"{gs.best_params_}"
                )

            # -----------------------------
            # TRAIN MODEL
            # -----------------------------

            model.fit(
                x_train,
                y_train
            )

            # -----------------------------
            # PREDICTIONS
            # -----------------------------

            y_train_pred = model.predict(
                x_train
            )

            y_test_pred = model.predict(
                x_test
            )

            # -----------------------------
            # R2 SCORE
            # -----------------------------

            train_model_score = r2_score(
                y_train,
                y_train_pred
            )

            test_model_score = r2_score(
                y_test,
                y_test_pred
            )

            print(
                f"Train R2 Score: "
                f"{train_model_score:.4f}"
            )

            print(
                f"Test R2 Score: "
                f"{test_model_score:.4f}"
            )

            # Store test score
            report[model_name] = test_model_score

            # IMPORTANT:
            # Store tuned model back into dictionary
            models[model_name] = model

        return report
    

    except Exception as e:

        raise CustomException(e, sys)
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)