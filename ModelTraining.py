import os
import pickle
import pandas as pd
from Data_PreProcessing import preprocess_data
from src.logger import logging
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier




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


x_train,x_test,y_train,y_test,preprocessor = preprocess_data(train_data,test_data,target_col,num_features,cat_features)

def encode(target):
    mapping = {
        "No":0,
        "Yes":1
    }
    return target.map(mapping)

y_train = encode(y_train)
y_test = encode(y_test)


class TrainingModels:
    def __init__(self):
        self.models = {
            "lr":LogisticRegression(
            max_iter=70,
            random_state=42),

            "knn":KNeighborsClassifier(leaf_size=30,n_jobs=5),

            "svm":SVC(probability=True,random_state=42),
            "dt":DecisionTreeClassifier(max_depth=20,min_samples_leaf=1,min_samples_split=2,random_state=42),
            "rrc":RandomForestClassifier(max_depth=15,min_samples_leaf=1,min_samples_split=2,random_state=42),
            "xb":XGBClassifier(random_state=42),
            "lgbm": LGBMClassifier(verbosity=-1,random_state=42),
            "cat":CatBoostClassifier(verbose=False,random_state=42)
        }

    def train_model(self,x_train,y_train):
        try:
            logging.info("starting baseline model training...")
            trained_models = {}
            for model_name,model in self.models.items():
                logging.info(f"training {model_name}")
                model.fit(x_train,y_train)
                trained_models[model_name] = model
                logging.info(f"{model_name} training complete")

            return trained_models
            logging.info("all baseline models trained")

        except Exception as e:
            logging.error(f"error occured during model training {e}")


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


