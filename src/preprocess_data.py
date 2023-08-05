import pandas as pd
from initialDataPreprocessing import InitialPreprocessingSteps
from save_dataset import save_data_to_csv
from completeDataPreprocessing import Complete_preprocessing




class PreprocessData:
    """
        Removes links,html tags,multiple spaces and expands contracted words for a given dataset in initial_preprocess
        Removes stopwords,punctuationsetc in complete preprocess

        data_path:Path to dataset

        Written By-Kanish Shandilya
    """
        

    def initial_preprocess(self,data_path,preprocess_path):
        try:
            print("Reading data from {}".format(data_path))
            dataset=pd.read_csv(data_path)
            print("Data Read successfully")
            preprocessingSteps=InitialPreprocessingSteps()
            print("Calling InitialPerprocessingSteps for preprocessing")
            preprocessed_dataset=preprocessingSteps.preprocessData(dataset)
            print("InitialPreprocessing finished")
            print("Saving dataset to {}".format(preprocess_path))
            saveFile=save_data_to_csv(preprocessed_dataset,preprocess_path)
            saveFile.save()
            print("Saved dataset to {}.".format(preprocess_path))
        
        except Exception as e:
            print("Exception occred at PreprocessData in method initial_preprocess {}".format(e))

    def complete_preprocess(self,dataset_path,preprocess_path):
        try:
            print("Reading data from {}".format(dataset_path))
            dataset=pd.read_csv(dataset_path)
            print("Data Read successfully")
            complete_preprocessingSteps=Complete_preprocessing()
            print("Calling CompletePerprocessingSteps for preprocessing")
            preprocessed_dataset=complete_preprocessingSteps.preprocess(dataset)
            print("Complete Preprocessing finished")
            print("Saving dataset to {}".format(preprocess_path))
            saveFile=save_data_to_csv(preprocessed_dataset,preprocess_path)
            saveFile.save()
            print("Saved dataset to {}.".format(preprocess_path))
        
        except Exception as e:
            print("Exception occred at complete PreprocessData in method preprocess {}".format(e))

