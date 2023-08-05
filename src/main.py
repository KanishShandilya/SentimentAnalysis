from load_dataset import load_dataset
import os
from save_dataset import save_data_to_csv
from split_dataset import Split_Dataset
from preprocess_data import PreprocessData
from tfidf_vectorizer import TFIDF_Vectorizer
from training import Train
from prepare_data_tf import PrepareData
from train_tfmodel import TrainTFModel
import pandas as pd
from read_json import ReadJson
class StartTraining:


    def start(self):
        read_json_obj=ReadJson("../config.json")

        original_dataset_path=read_json_obj.read_attribute("original_dataset_path")

        transformed_data_filename=read_json_obj.read_attribute("transformed_data_fileName")

        transformed_dataset_path=read_json_obj.read_attribute("transformed_dataset_path")

        train_data_path=read_json_obj.read_attribute("train_data")

        test_data_path=read_json_obj.read_attribute("test_data")

        train_data_for_preprocessing=read_json_obj.read_attribute("train_data_for_preprocessing")

        test_data_for_preprocessing=read_json_obj.read_attribute("test_data_for_preprocessing")

        train_data_initialPreprocess=read_json_obj.read_attribute("train_data_initialPreprocess")

        test_data_initialPreprocess=read_json_obj.read_attribute("test_data_initialPreprocess")

        train_data_completePreprocess=read_json_obj.read_attribute("train_data_completePreprocess")

        test_data_completePreprocess=read_json_obj.read_attribute("test_data_completePreprocess")

        tfidf_saved_object=read_json_obj.read_attribute("tfidf_obj_save")

        y_train_path=read_json_obj.read_attribute("y_train_path")

        y_test_path=read_json_obj.read_attribute("y_test_path")


        transformed_dataFile=os.path.join(transformed_dataset_path,transformed_data_filename)
        
        #Load original data
        load_data_obj=load_dataset(original_dataset_path)
        df=load_data_obj.read_data()
        if not os.path.exists(transformed_dataset_path):
            os.mkdir(transformed_dataset_path)
        #save tranformed data
        save_data_to_csv(df,transformed_dataFile).save()
        #split data
        split_data=Split_Dataset(transformed_dataFile,train_data_path=train_data_path,test_data_path=test_data_path)
        split_data.split()

        #preprocess data
        preprocess_obj=PreprocessData()

        preprocess_obj.initial_preprocess(train_data_for_preprocessing,train_data_initialPreprocess)
        preprocess_obj.initial_preprocess(test_data_for_preprocessing,test_data_initialPreprocess)


        preprocess_obj.complete_preprocess(train_data_initialPreprocess,train_data_completePreprocess)
        preprocess_obj.complete_preprocess(test_data_initialPreprocess,test_data_completePreprocess)

        #ML models
        tfidf_obj=TFIDF_Vectorizer(train_data_completePreprocess,ngram_range=(1,2),read_json_obj=read_json_obj)
        X_train=tfidf_obj.get_features()
        X_test=tfidf_obj.transform(tfidf_saved_object,test_data_completePreprocess)

        y_train=pd.read_csv(y_train_path)
        y_train=y_train["Score"].values
        y_test=pd.read_csv(y_test_path)
        y_test=y_test["Score"].values

        train=Train(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,read_json_obj=read_json_obj)
        train.train()


        #train_tfmodel_obj=TrainTFModel(data_train=train_data_initialPreprocess,data_test=test_data_initialPreprocess,y_train=y_train,y_test=y_test,read_json_obj=read_json_obj)
        #train_tfmodel_obj.train()





if __name__=="__main__":
    obj=StartTraining()
    obj.start()
        

        