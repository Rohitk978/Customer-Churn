import os
from src.logger import logging
import pickle
import pandas as pd
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score)

from Data_PreProcessing import preprocess_data

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

def encode(target):
    mapping = {"No":0,"Yes":1}
    return target.map(mapping)

x_train,x_test,y_train,y_test,preprocesser = preprocess_data(train_data,test_data,target_col,num_features,cat_features)
y_test = encode(y_test)


class modelcomparision:
    def __init__(self):
        logging.info("Models comparision initialized")

    def load_models(self, model_dir):
        try:
            logging.info(f"Loading models from {model_dir}")
            trained_models = {}

            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"Directory does not exist: {model_dir}")
            

            for model_file in sorted(os.listdir(model_dir)):
                if model_file.endswith(".pkl"):
                    model_path = os.path.join(model_dir,model_file)

                    with open(model_path,"rb") as file:
                        model = pickle.load(file)

                    model_name = os.path.splitext(model_file)[0]
                trained_models[model_name] = model
                logging.info(f"{model_name} model loaded successfully")

            logging.info(f"all models loaded from {model_dir}")
            return trained_models
        except Exception as e:
            logging.error(f"error occured during loading model {e}")
            raise

    def evaluate_models(self,models,x_test,y_test,model_type):
        try:
            logging.info(f"starting evaluation of {model_type} models")

            results = []

            for model_name,model in models.items():
                logging.info(f"evaluating {model_name} {model_type} model")

                y_pred = model.predict(x_test)
                accuracy =  accuracy_score(y_test,y_pred)

                precision = precision_score(y_test,y_pred,pos_label=1,zero_division=0)

                recall = recall_score(y_test,y_pred,pos_label=1,zero_division=0)

                f1 = f1_score(y_test,y_pred,pos_label=1,zero_division=0)

                results.append({
                    "model_name":model_name,
                    "model_type":model_type,
                    "accuracy":accuracy,
                    "precision":precision,
                    "recall":recall,
                    "f1_Score":f1
                })

                logging.info(f"evaluation completed {model_type}")

            results_df = pd.DataFrame(results)
            logging.info(f"{model_type} models evaluation completed")

            return results_df
        except Exception as e:
            logging.error(f"error occured during models evaluation {e}")
            raise


