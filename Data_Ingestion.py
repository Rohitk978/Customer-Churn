import pandas as pd
from pathlib import Path
from pydantic import BaseModel,FilePath, field_validator
from sklearn.model_selection import train_test_split
from src.logger import logger



class DtatIngestion(BaseModel):
    """
    Here we cleaning the file data Not transforming the data because if we do that then it would also transform the test data which reflects the real world data or unseen data.
    And splitting the data into train.csv and test.csv file.
    Saving those file for further use.

    """

    

    def load_dataset(self,file_path):
        logger.info("Loading dataset")
        try:
            df = pd.read_csv(file_path)
            logger.info("dataset loaded successfully")
            logger.info(f"dataset shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Unalbe to load dataset{e}")

    def cleaning(self,df:pd.DataFrame) -> pd.DataFrame:
        logger.info("cleaning the dataset")
        df = df.drop_duplicates()
        df = df.dropna(how="all")

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ","_")
        )

        print(df.columns)

        object_col = df.select_dtypes(include="object").columns
        for col in object_col:
            df[col] = df[col].str.strip()
        if "totalchages" in df.columns:
            df['totalcharges'] = pd.to_numeric(
                df['totalcharges'],
                error = "coerce"
            )
        df.reset_index(drop=True,inplace=True)

        return df

    def splitdataset(self,df,target_column='churn',test_size=0.2,random_state=32):
        logger.info("splitting the dataset into train and test")
        try:
            logger.info(f"dataset {df.head()}")

            logger.info(f"Type of df: {type(df)}")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            logger.info(f"Missing values:\n{df.isnull().sum()}")
            logger.info(f"Target Distribution:\n{df[target_column].value_counts()}")

            train_df, test_df = train_test_split(
                df,test_size=test_size,random_state=random_state,stratify=df[target_column]
            )
            logger.info(f"train_df shape{train_df.shape}")
            logger.info(f"test_df shape{test_df.shape}")

            return train_df,test_df
        except Exception as e:
            logger.exception("dataset splitting failed")
            

    def savedata(self,train_df,test_df,output_dir="data"):
        logger.info("saving training and testing data files")
        try:
            output_path = Path(output_dir)
            output_path.mkdir(
                parents=True,
                exist_ok=True
            )

            train_path = output_path / "train.csv"
            test_path = output_path / "test.csv"

            logger.info(f"saving train dataset to {train_path}")
            train_df.to_csv(
                train_path,
                index=False
            )
            logger.info("successfully saved trained data")

            logger.info(f"saving test dataset to {test_path}")
            test_df.to_csv(
                test_path,
                index=False
            )
            logger.info("successfully saved tested data")


            return {
                "train_path": str(train_path),
                "test_path" : str(test_path)
            }
            logger.info("Saved the files.")

        except Exception as e:
            return {"messages":f"saving dataset failed {e}"}

    def initiate_dataingesttion(self,file_path):
        logger.info("starting data ingestion function")
        try:
            df = self.load_dataset(file_path)
            clean_df = self.cleaning(df)
            split_df = self.splitdataset(clean_df)
            train_df = split_df[0]
            test_df = split_df[1]
            saved_paths = self.savedata(
                train_df,
                test_df
            )
            logger.info("everythhing went well.")
        except Exception as e:
            return{"messages":f"Data ingestion failed {e}"}

