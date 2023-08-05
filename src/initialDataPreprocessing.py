import re
from save_dataset import save_data_to_csv
import pandas
import string





class InitialPreprocessingSteps:
    """
        This clas removes urls,link,multipleSpaces and return dataset

        Written By-Kanish Shandilya
    """

    def __init__(self):
        
        self.html_regex=r'<.*?>'
        self.url_regex=r"http\S+"

    def expand_text(self,text):
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        return text
    
    def remove_url(self,text:str):
        return re.sub(self.url_regex, "", text)

    def remove_htmlTags(self,text:str):
        text = re.sub(self.html_regex, '', text)
        return text
    def removeExtra(self,text:str):
        text=re.sub(r'\n', ' ', text)
        text=re.sub(r'-', ' ', text)
        text = re.sub(r'\([A-za-z\d]*\)', '', text)
        text = re.sub(r" +", ' ', text)
        text = text.strip()
        text="".join([i for i in text if i not in string.punctuation])
        text=text.lower()
        return text
    def preprocessData(self,dataset:pandas.Series):
        """
            dataset-Dataset to preprocess

            Written By- Kanish Shandilya

        """
        preprocessed_dataPath="../preprocessed_data/reviews_preprocessed.csv"
        print("Starting data preprocessing")
        print("Executing expand_text")
        dataset["transformed_text"]=dataset["review_text"].map(lambda x:self.expand_text(x))
        print("expand_text completed. Removing Html tags")
        dataset["transformed_text"]=dataset["transformed_text"].map(lambda x:self.remove_htmlTags(x))
        print("HTML tags removed. Removing url")
        dataset["transformed_text"]=dataset["transformed_text"].map(lambda x:self.remove_url(x))
        print("URL removed. Removing multiple spaces")
        dataset["transformed_text"]=dataset["transformed_text"].map(lambda x:self.removeExtra(x))
        print("Data preprocessing completed. Droping review_text column")
        dataset=dataset.drop(["review_text"],axis=1)
        print("Dropped review_text column.")
        return dataset
