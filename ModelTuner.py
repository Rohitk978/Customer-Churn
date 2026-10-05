import optuna
import pandas as pd 
from src.logger import logging
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from Data_PreProcessing import preprocess_data
import pickle
import os

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
target_col = "churn"
num_features = [
    "tenure",
    "monthlycharges",
    "totalcharges"
]
cat_features = [
    "gender",
    "partner",
    "dependents",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod"
]

x_train,x_test,y_train,y_test,preprocesser = preprocess_data(train_data,test_data,target_col,num_features,cat_features)

def encode(target):
    mapping = {"No":0,"Yes":1}
    return target.map(mapping)


y_train = encode(y_train)

class ModelTuning:
    def __init__(self):
        logging.info("model tuner initialization.")

    def cross_validation_score(self,model,x_train,y_train):
        try:
            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            )
            scores = []

            for fold,(train_index,valid_index) in enumerate(cv.split(x_train,y_train),start=1):
                logging.info(f"starting fold {fold}")

                x_fold_train = x_train[train_index]
                x_fold_valid = x_train[valid_index]

                y_fold_train = y_train[train_index]
                y_fold_valid = y_train[valid_index]

                model.fit(x_fold_train,y_fold_train)
                logging.info(f"model fitted for fold {fold}")

                y_pred = model.predict(x_fold_valid)

                score = f1_score(y_fold_valid,y_pred)
                scores.append(score)
                logging.info(f"fold {fold} f1_score {score}")

            mean_score = sum(scores) / len(scores)
            logging.info(f"mean f1 score : {mean_score}")

            return mean_score

        except Exception as e:
            logging.error(f"error during cross validation {e}")
            raise


    
    def objective_logistic_regression(
            self,
            trial,
            X_train,
            y_train
    ):

        C = trial.suggest_float(
            "C",
            0.001,
            20,
            log=True
        )

        solver = trial.suggest_categorical(
            "solver",
            [
                "liblinear",
                "lbfgs"
            ]
        )

        model = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=50,
            random_state=42
        )

        score = self.cross_validation_score(
            model,
            X_train,
            y_train
        )

        return score

    def tune_logistic_regression(
            self,
            X_train,
            y_train,
            n_trials=20
    ):

        logging.info(
            "Starting Logistic Regression tuning"
        )

        study = optuna.create_study(
            direction="maximize",
            study_name="logistic_regression_optimization"
        )

        study.optimize(
            lambda trial: self.objective_logistic_regression(
                trial,
                X_train,
                y_train
            ),
            n_trials=n_trials
        )

        logging.info(
            f"Best Logistic Regression F1: "
            f"{study.best_value}"
        )

        logging.info(
            f"Best Logistic Regression parameters: "
            f"{study.best_params}"
        )

        return study

    # DECISION TREE

    def objective_decision_tree(
            self,
            trial,
            X_train,
            y_train
    ):

        criterion = trial.suggest_categorical(
            "criterion",
            [
                "gini",
                "entropy"
            ]
        )

        max_depth = trial.suggest_int(
            "max_depth",
            2,
            20
        )

        min_samples_split = trial.suggest_int(
            "min_samples_split",
            2,
            20
        )

        min_samples_leaf = trial.suggest_int(
            "min_samples_leaf",
            1,
            10
        )

        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

        score = self.cross_validation_score(
            model,
            X_train,
            y_train
        )

        return score

    def tune_decision_tree(
            self,
            X_train,
            y_train,
            n_trials=20
    ):

        logging.info(
            "Starting Decision Tree tuning"
        )

        study = optuna.create_study(
            direction="maximize",
            study_name="decision_tree_optimization"
        )

        study.optimize(
            lambda trial: self.objective_decision_tree(
                trial,
                X_train,
                y_train
            ),
            n_trials=n_trials
        )

        logging.info(
            f"Best Decision Tree F1: "
            f"{study.best_value}"
        )

        logging.info(
            f"Best Decision Tree parameters: "
            f"{study.best_params}"
        )

        return study

    # SVM

    def objective_svm(
            self,
            trial,
            X_train,
            y_train
    ):

        C = trial.suggest_float(
            "C",
            0.01,
            50,
            log=True
        )

        kernel = trial.suggest_categorical(
            "kernel",
            [
                "linear",
                "rbf",
                "poly"
            ]
        )

        gamma = trial.suggest_categorical(
            "gamma",
            [
                "scale",
                "auto"
            ]
        )

        model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            random_state=42
        )

        score = self.cross_validation_score(
            model,
            X_train,
            y_train
        )

        return score

    def tune_svm(
            self,
            X_train,
            y_train,
            n_trials=20
    ):

        logging.info(
            "Starting SVM tuning"
        )

        study = optuna.create_study(
            direction="maximize",
            study_name="svm_optimization"
        )

        study.optimize(
            lambda trial: self.objective_svm(
                trial,
                X_train,
                y_train
            ),
            n_trials=n_trials
        )

        logging.info(
            f"Best SVM F1: "
            f"{study.best_value}"
        )

        logging.info(
            f"Best SVM parameters: "
            f"{study.best_params}"
        )

        return study

    # RANDOM FOREST

    def objective_random_forest(
            self,
            trial,
            X_train,
            y_train
    ):

        n_estimators = trial.suggest_int(
            "n_estimators",
            20,
            50
        )

        max_depth = trial.suggest_int(
            "max_depth",
            3,
            20
        )

        min_samples_split = trial.suggest_int(
            "min_samples_split",
            2,
            20
        )

        min_samples_leaf = trial.suggest_int(
            "min_samples_leaf",
            1,
            10
        )

        max_features = trial.suggest_categorical(
            "max_features",
            [
                "sqrt",
                "log2"
            ]
        )

        criterion = trial.suggest_categorical(
            "criterion",
            [
                "gini",
                "entropy"
            ]
        )

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            random_state=42,
            n_jobs=-1
        )

        score = self.cross_validation_score(
            model,
            X_train,
            y_train
        )

        return score

    def tune_random_forest(
            self,
            X_train,
            y_train,
            n_trials=20
    ):

        logging.info(
            "Starting Random Forest tuning"
        )

        study = optuna.create_study(
            direction="maximize",
            study_name="random_forest_optimization"
        )

        study.optimize(
            lambda trial: self.objective_random_forest(
                trial,
                X_train,
                y_train
            ),
            n_trials=n_trials
        )

        logging.info(
            f"Best Random Forest F1: "
            f"{study.best_value}"
        )

        logging.info(
            f"Best Random Forest parameters: "
            f"{study.best_params}"
        )

        return study

    # XGBOOST

    def objective_xgboost(
            self,
            trial,
            X_train,
            y_train
    ):

        n_estimators = trial.suggest_int(
            "n_estimators",
            100,
            500
        )

        learning_rate = trial.suggest_float(
            "learning_rate",
            0.01,
            0.3,
            log=True
        )

        max_depth = trial.suggest_int(
            "max_depth",
            3,
            12
        )

        min_child_weight = trial.suggest_int(
            "min_child_weight",
            1,
            10
        )

        subsample = trial.suggest_float(
            "subsample",
            0.6,
            1.0
        )

        colsample_bytree = trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        )

        gamma = trial.suggest_float(
            "gamma",
            0,
            5
        )

        reg_alpha = trial.suggest_float(
            "reg_alpha",
            1e-8,
            10,
            log=True
        )

        reg_lambda = trial.suggest_float(
            "reg_lambda",
            1e-8,
            10,
            log=True
        )

        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        )

        score = self.cross_validation_score(
            model,
            X_train,
            y_train
        )

        return score

    def tune_xgboost(
            self,
            X_train,
            y_train,
            n_trials=50
    ):

        logging.info(
            "Starting XGBoost tuning"
        )

        study = optuna.create_study(
            direction="maximize",
            study_name="xgboost_optimization"
        )

        study.optimize(
            lambda trial: self.objective_xgboost(
                trial,
                X_train,
                y_train
            ),
            n_trials=n_trials
        )

        logging.info(
            f"Best XGBoost F1: "
            f"{study.best_value}"
        )

        logging.info(
            f"Best XGBoost parameters: "
            f"{study.best_params}"
        )

        return study

    # LIGHTGBM

    def objective_lightgbm(
            self,
            trial,
            X_train,
            y_train
    ):

        n_estimators = trial.suggest_int(
            "n_estimators",
            100,
            500
        )

        learning_rate = trial.suggest_float(
            "learning_rate",
            0.01,
            0.3,
            log=True
        )

        num_leaves = trial.suggest_int(
            "num_leaves",
            10,
            100
        )

        max_depth = trial.suggest_int(
            "max_depth",
            -1,
            20
        )

        min_child_samples = trial.suggest_int(
            "min_child_samples",
            5,
            50
        )

        subsample = trial.suggest_float(
            "subsample",
            0.6,
            1.0
        )

        colsample_bytree = trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        )

        reg_alpha = trial.suggest_float(
            "reg_alpha",
            1e-8,
            10,
            log=True
        )

        reg_lambda = trial.suggest_float(
            "reg_lambda",
            1e-8,
            10,
            log=True
        )

        model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

        score = self.cross_validation_score(
            model,
            X_train,
            y_train
        )

        return score

    def tune_lightgbm(
            self,
            X_train,
            y_train,
            n_trials=50
    ):

        logging.info(
            "Starting LightGBM tuning"
        )

        study = optuna.create_study(
            direction="maximize",
            study_name="lightgbm_optimization"
        )

        study.optimize(
            lambda trial: self.objective_lightgbm(
                trial,
                X_train,
                y_train
            ),
            n_trials=n_trials
        )

        logging.info(
            f"Best LightGBM F1: "
            f"{study.best_value}"
        )

        logging.info(
            f"Best LightGBM parameters: "
            f"{study.best_params}"
        )

        return study

    # CATBOOST

    def objective_catboost(
            self,
            trial,
            X_train,
            y_train
    ):

        iterations = trial.suggest_int(
            "iterations",
            100,
            500
        )

        learning_rate = trial.suggest_float(
            "learning_rate",
            0.01,
            0.3,
            log=True
        )

        depth = trial.suggest_int(
            "depth",
            4,
            10
        )

        l2_leaf_reg = trial.suggest_float(
            "l2_leaf_reg",
            1,
            10
        )

        random_strength = trial.suggest_float(
            "random_strength",
            0,
            5
        )

        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_strength=random_strength,
            random_state=42,
            verbose=False
        )

        score = self.cross_validation_score(
            model,
            X_train,
            y_train
        )

        return score

    def tune_catboost(
            self,
            X_train,
            y_train,
            n_trials=50
    ):

        logging.info(
            "Starting CatBoost tuning"
        )

        study = optuna.create_study(
            direction="maximize",
            study_name="catboost_optimization"
        )

        study.optimize(
            lambda trial: self.objective_catboost(
                trial,
                X_train,
                y_train
            ),
            n_trials=n_trials
        )

        logging.info(
            f"Best CatBoost F1: "
            f"{study.best_value}"
        )

        logging.info(
            f"Best CatBoost parameters: "
            f"{study.best_params}"
        )

        return study



    def train_best_model(self,model_name,study,x_train,y_train):
        try:
            best_params = study.best_params
            logging.info(f"training final tuned model" f"{model_name}")

            if model_name == "logistic_regression":

                model = LogisticRegression(
                    **best_params,
                    max_iter=1000,
                    random_state=42
                )

            elif model_name == "decision_tree":

                model = DecisionTreeClassifier(
                    **best_params,
                    random_state=42
                )

            elif model_name == "svm":

                model = SVC(
                    **best_params,
                    random_state=42
                )

            elif model_name == "random_forest":

                model = RandomForestClassifier(
                    **best_params,
                    random_state=42,
                    n_jobs=-1
                )

            elif model_name == "xgboost":

                model = XGBClassifier(
                    **best_params,
                    random_state=42,
                    eval_metric="logloss",
                    n_jobs=-1
                )

            elif model_name == "lightgbm":

                model = LGBMClassifier(
                    **best_params,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1
                )

            elif model_name == "catboost":

                model = CatBoostClassifier(
                    **best_params,
                    random_state=42,
                    verbose=False
                )

            else:

                raise ValueError(
                    f"Unsupported model: {model_name}"
                )

            # FINAL FIT

            model.fit(
                x_train,
                y_train
            )

            logging.info(
                f"{model_name} final tuned model "
                f"fitted successfully"
            )

            return model

        except Exception as e:

            logging.error(
                f"Error training best model "
                f"{model_name}: {e}"
            )


    def tune_all_models(
            self,
            X_train,
            y_train,
            n_trials=20
         ):

        try:

            studies = {}

            logging.info(
                "Starting tuning of all models"
            )

            studies["logistic_regression"] = (
                self.tune_logistic_regression(
                    X_train,
                    y_train,
                    n_trials
                )
            )

            studies["decision_tree"] = (
                self.tune_decision_tree(
                    X_train,
                    y_train,
                    n_trials
                )
            )

            studies["svm"] = (
                self.tune_svm(
                    X_train,
                    y_train,
                    n_trials
                )
            )

            studies["random_forest"] = (
                self.tune_random_forest(
                    X_train,
                    y_train,
                    n_trials
                )
            )

            studies["xgboost"] = (
                self.tune_xgboost(
                    X_train,
                    y_train,
                    n_trials
                )
            )

            studies["lightgbm"] = (
                self.tune_lightgbm(
                    X_train,
                    y_train,
                    n_trials
                )
            )

            studies["catboost"] = (
                self.tune_catboost(
                    X_train,
                    y_train,
                    n_trials
                )
            )

            tuned_models = {}

            for model_name, study in studies.items():

                tuned_models[model_name] = (
                    self.train_best_model(
                        model_name,
                        study,
                        X_train,
                        y_train
                    )
                )

            logging.info(
                "All models tuned and trained successfully"
            )

            return tuned_models, studies

        except Exception as e:

            logging.error(
                f"Error during tuning all models: {e}"
            )



    def save_models(self,trained_models,model_dir):
            try:
    
                logging.info("mdoel saving started")
    
                os.makedirs(
                    model_dir,exist_ok=True
                )
    
                for model_name, models in trained_models.items():
                    model_path = os.path.join(
                        model_dir,
                        f"{model_name}.pkl"
                    )
    
                    with open(model_path,"wb") as file:
                        pickle.dump(models,file)
                    logging.info(f"{model_name} saved at {model_path}")
    
                logging.info("all baseline models saved")
    
            except Exception as e:
                logging.error(f"error occured during saving models {e}")

        
                
