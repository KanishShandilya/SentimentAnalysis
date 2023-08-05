import pandas as pd

class loadData:
  """
  Load the Dataset & Drops columns

  dataset_path: Path to reviews.csv

  Written By- Kanish Shandilya
  """

  def __init__(self,dataset_path):
    self.dataset_path=dataset_path
    #############################
    self.drop_column_list=["Time","ProductId","UserId","Id","ProfileName","HelpfulnessNumerator","HelpfulnessDenominator"]

  def read_data(self):
    try:
      print("Starting dataset reading from {}".format(self.dataset_path))
      dataset=pd.read_csv(self.dataset_path).iloc[:175000,:]
      print("Read data from {}.Dropping columns now.".format(self.dataset_path))
      dataset=self.drop_columns(dataset)
      print("Successfully dropped columns. Now merging Text and Summary Column")
      dataset=self.mergeTextAndSummary(dataset)
      print("Successfully merged Text and Summary Column.")
      print("Now Exiting Function")
      return dataset
    except FileNotFoundError:
      print("No such file exists at {}".format(self.dataset_path))
    except Exception as e:
      print("Exception occured at load_dataset read_data method {}".format(e))
  
  def drop_columns(self,dataset):
    try:
      print("Dropping following columns from dataset {} ",self.drop_column_list)
      return dataset.drop(self.drop_column_list,axis=1)
    except Exception as e:
      print("Exception occured at load_dataset drop_columns ",e)

  def mergeTextAndSummary(self,dataset):
    try:
      dataset["review_text"]=dataset["Summary"].astype(str)+" "+dataset["Text"]
      return dataset.drop(["Summary",'Text'],axis=1)
    except Exception as e:
      print("Exception occured at load_dataset mergeTextAndSummary function {}".format(e))