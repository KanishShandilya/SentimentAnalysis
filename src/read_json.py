import json



class ReadJson:
    """
        Read data from json
    """
    def __init__(self,json_filePath):
        self.json_filePath=json_filePath

    def read_attribute(self,att_name="original_dataset_path"):
        f=open(self.json_filePath)

        data=json.load(f)

        return data[att_name]
