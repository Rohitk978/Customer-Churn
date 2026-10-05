from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import os
import pickle
from src.logger import logging


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


def create_num_pipeline(num_columns):
    numerical_pipeline = Pipeline(
        steps=[
            ("Imputer",SimpleImputer(strategy='median')),
            ("scaler",StandardScaler())
        ]
    )

    return numerical_pipeline


def create_cat_pipeline(cat_columns):
    categorical_pipeline = Pipeline(
        steps=[
            ("impute",SimpleImputer(strategy='most_frequent')),
            ("encoder",OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    return categorical_pipeline


def create_preprocessor(num_columns,cat_columns):
    numerical_pipeline = create_num_pipeline(num_columns)
    categorical_pipeline = create_cat_pipeline(cat_columns)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",numerical_pipeline,num_columns
            ),
            (
                "categorical",categorical_pipeline,cat_columns
            )
        ],remainder="drop"
    )

    return preprocessor







train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
target_col = "churn"

def preprocess_data(train_data,test_data,target_col,num_features,cat_features,id_columns="customerid"):
    x_train = train_data.drop(target_col,axis=1)
    y_train = train_data[target_col]

    x_test = test_data.drop(target_col,axis=1)
    y_test = test_data[target_col]

    if id_columns in x_train.columns:
        x_train = x_train.drop(id_columns,axis=1)
    if id_columns in x_test.columns:
        x_test = x_test.drop(id_columns,axis=1)

   

    preprocessor = create_preprocessor(
        num_features,
        cat_features
    )

    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)

    return (
        x_train_processed,
        x_test_processed,
        y_train,
        y_test,
        preprocessor
    )



def save_preprocessor(preprocessor,preprocessor_path):
    try:
        directory = os.path.dirname(preprocessor_path)
        if directory:
            os.makerdirs(directory,exist_ok=True)
        with open(preprocessor_path,"wb") as file:
            pickle.dump(preprocessor,file)

        logging.info(f"preprocessor saved successfully saved {preprocessor_path }")

        

    except Exception as e:
        raise(f"Error {e}")


def load_preprocessor(preprocessor_path):
    try:
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"preprocessor not found {preprocessor_path}")
        with open(preprocessor_path,"rb") as file:
            preprocessor = pickle.load(file)
        return preprocessor

    except Exception as e:
        raise RuntimeError(f"failed to load preprocessor {e}")

def preprocess_user_input_data(input_data,preprocessor,id_columns="customerid"):
    if isinstance(input_data,dict):
        input_data  = pd.DataFrame([input_data])
    elif not isinstance(input_data,pd.DataFrame):
        raise TypeError(
            "Input Invalid!!!"
        )

    # we drop the customer id from main dataset so we doing the same in this if it have any.
    if id_columns in input_data.columns:
        input_data = input_data.drop(id_columns,axis=1)

    # num_data = input_data.select_dtypes(include=["int64","float64"])
    # cat_data = input_data.select_dtypes(include=["object","category"])

    # preprocessor = create_preprocessor(num_data,cat_data)

    input_data_processed = (preprocessor.transform(input_data))
    return input_data_processed




# pth = save_preprocessor(preprocessor,"tunemodels/")