# import pandas as pd 
# from Data_PreProcessing import preprocess_user_input_data
# import pickle
# from src.logger import logging
# import os

# class PredictionPipeline:
#     def __init__(self,model_path,preprocessor_path):
#         self.model_path = model_path
#         self.preprocessor_path = preprocessor_path
#         self.model = None
#         self.preprocessor = None
#         logging.info("prediction pipeline started....")

#     def load_model(self):
#         try:
#             if not  os.path.exists(self.model_path):
#                 raise FileNotFoundError(f"model not found {self.model_path}")
#             with open(self.model_path,"rb") as file:
#                 self.model = pickle.load(file)
#             logging.info("Model successfully loaded.....")
#         except Exception as e:
#             raise(f"Model cannot successfully loaded...{e}")

#     def load_preprocessor(self,input_data):
#         try:
#             logging.info("preprocessing started ......")
#             if not os.path.exists(self.preprocessor_path):
#                 raise FileNotFoundError(f"file not found {self.preprocessor_path}")
#             with open(self.preprocessor_path,"rb") as file:
#                 self.preprocessor = pickle.load(file)
#             logging.info("preprocessor loaded.......")

#         except Exception as e:
#             raise(f"preprocessor cannot be loaded... {e}")

#     def preprocess_input(self,input_data):
#         try:
#             logging.info("preprocessing input started....")
#             preprocess_data = (preprocess_user_input_data(input_data,self.preprocessor))
#             logging.info(f"user processing completed {preprocess_data.shape}")
#             return preprocess_data

#         except Exception as e:
#             logging.error(f"error during preprocessing {e}")
        

#     def predict(self,input_data):
#         logging.info("predicting.......")
#         try:
#             self.load_model()
#             predata = self.load_preprocessor(input_data)
#             processed_data = (self.preprocess_input(predata))
#             prediction = (self.model.predict(processed_data))
#             probability = (self.model.predict_proba(processed_data))
#             prediction_value = int(prediction[0])
#             churn_probability = float(probability[0][1])
#             if prediction_value == 1:
#                 result = ("Customer is likely to churn")
#             else:
#                 result = ("Customer is unlikely to churn")

#             logging.info("preidicting completed.....")
#             return {
#                 "prediction":prediction_value,
#                 "result":result,
#                 "probability":churn_probability
#             }

#         except Exception as e:
#             raise(f"Couldn't predict output!!")







# NEW CODE.


import pickle
import os
from Data_PreProcessing import preprocess_user_input_data
from src.logger import logging

class PredictionPipeline:
    def __init__(self, model_path, preprocessor_path):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        logging.info("Prediction pipeline started")

    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            with open(self.model_path, "rb") as file:
                self.model = pickle.load(file)
            logging.info("Model successfully loaded")
            return self.model
        except Exception as e:
            logging.error(f"Model loading failed: {e}", exc_info=True)
            raise RuntimeError(f"Model could not be loaded: {e}") from e

    def load_preprocessor(self):
        try:
            if not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError(f"Preprocessor not found: {self.preprocessor_path}")
            with open(self.preprocessor_path, "rb") as file:
                self.preprocessor = pickle.load(file)
            logging.info("Preprocessor successfully loaded")
            return self.preprocessor
        except Exception as e:
            logging.error(f"Preprocessor loading failed: {e}", exc_info=True)
            raise RuntimeError(f"Preprocessor could not be loaded: {e}") from e

    def preprocess_input(self, input_data):
        try:
            logging.info("User input preprocessing started")
            processed_data = preprocess_user_input_data(
                input_data,
                self.preprocessor
            )
            if processed_data is None:
                raise ValueError("preprocess_user_input_data returned None")
            logging.info(f"User preprocessing completed. Shape: {processed_data.shape}")
            return processed_data
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}", exc_info=True)
            raise

    def predict(self, input_data):
        try:
            logging.info("Prediction started")
            self.load_model()
            self.load_preprocessor()
            processed_data = self.preprocess_input(input_data)
            prediction = self.model.predict(processed_data)
            probability = self.model.predict_proba(processed_data)
            prediction_value = int(prediction[0])
            churn_probability = float(probability[0][1])
            result = "Customer is likely to churn" if prediction_value == 1 else "Customer is unlikely to churn"
            logging.info("Prediction completed successfully")
            return {
                "prediction": prediction_value,
                "result": result,
                "probability": churn_probability
            }
        except Exception as e:
            logging.error(f"Prediction failed: {e}", exc_info=True)
            raise


    
