import pandas as pd


class save_data_to_csv:
  """
  Save given dataset to given path

  dataset:Dataset to be saved
  path:path to be saved to

  Written By-Kanish Shandilya

  """
  
  def __init__(self,dataset,path):
    self.dataset=dataset
    self.path=path

  def save(self):
    try:
      print("Saving file at {}".format(self.path))
      self.dataset.to_csv(self.path,index=False)
    except Exception as e:
      print('Error occured at save_data_to_csv save method {}'.format(e))