from loadData import loadData

from src.preprocess_data import PreprocessData

class Predict:
    """
        Given a csv files it gives prediction of their sentiment

        Written By- Kanish Shandilya
    """
    def __init__(self,dataset_path):
        """
            dataset_path: where data to be predicted is stored
        """
        self.dataset_path=dataset_path

    def predict(self):
        loadDataobj=loadData(self.dataset_path)
        dataset=loadDataobj.read_data()
        preprocess_data=PreprocessData()
        preprocess_obj.initial_preprocess()