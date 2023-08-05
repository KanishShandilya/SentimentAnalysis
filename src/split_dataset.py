import pandas as pd
from sklearn.model_selection import train_test_split
from save_dataset import save_data_to_csv
import os

class Split_Dataset:
    """
    Split the transformed dataset for train and test and saves data to given path

    transformed_dataset_path-Path to dataset
    train_data_path-Path to save train_data
    test_data_path-Path to save test data

    Written By- Kanish Shandilya
    """

    def __init__(self,transformed_data_path:str,train_data_path:str,test_data_path:str):
        self.transformed_data_path=transformed_data_path
        self.train_data_path=train_data_path
        self.test_data_path=test_data_path
    
    def split(self,test_size=0.2)->None:
        try:
            print("Reading transformed dataset from {}".format(self.transformed_data_path))
            df=pd.read_csv(self.transformed_data_path)
            print("Read data successfully. Now seperating label and text")
            X=df["review_text"]
            y=df["Score"]
            print("Operation successful. Now spliting data to train and test")
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size)
            print("Successful splited data to train and test. Now saving files to {} and {}".format(self.train_data_path,self.test_data_path))
            print("Saving X_train to {}".format(os.path.join(self.train_data_path,"x_train.csv")))
            saveDataTrainX=save_data_to_csv(X_train,os.path.join(self.train_data_path,"x_train.csv"))
            saveDataTrainX.save()
            print("Saved X_train to {}".format(os.path.join(self.train_data_path,"x_train.csv")))
            print("Saving y_train to {}".format(os.path.join(self.train_data_path,"y_train.csv")))
            saveDataTrainY=save_data_to_csv(y_train,os.path.join(self.train_data_path,"y_train.csv"))
            saveDataTrainY.save()
            print("Saved y_train to {}".format(os.path.join(self.train_data_path,"y_train.csv")))
            print("Saving X_test to {}".format(os.path.join(self.test_data_path,"x_test.csv")))
            saveDataTestX=save_data_to_csv(X_test,os.path.join(self.test_data_path,"x_test.csv"))
            saveDataTestX.save()
            print("Saved X_test to {}".format(os.path.join(self.test_data_path,"x_test.csv")))
            print("Saving ytest to {}".format(os.path.join(self.test_data_path,"y_test.csv")))
            saveDataTestY=save_data_to_csv(y_test,os.path.join(self.test_data_path,"y_test.csv"))
            saveDataTestY.save()
            print("Saved y_test to {}".format(os.path.join(self.train_data_path,"y_test.csv")))

            print("Succesfully spited data and saved to file . Exiting function")
        except Exception as e:
            print("Exception occured at Split_Dataset split method {}".format(e))

        