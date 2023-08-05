import nltk
import re
from nltk.stem import WordNetLemmatizer
import string
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import PorterStemmer

class Complete_preprocessing:
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.tokenizer=WhitespaceTokenizer()
        self.lemmatizer=WordNetLemmatizer()
        self.stemmer=PorterStemmer()

    def tokenize(self,text,remove_stopwords:bool=True,lemmatize:bool=True,stem:bool=True):
        """
            text:Text to tokenize
            remove_stopwords:Whether to remove stopwords or not (default True)
            lemmatize:Whether to lemmatize or not (default True)
            stem:Whether to do stemming or not (default True)

            Written By- Kanish Shandilya
        """
        tokens = self.tokenizer.tokenize(text)
        if remove_stopwords:
            tokens= [i for i in tokens if i not in self.stopwords]
        if stem:
            tokens=[self.stemmer.stem(word) for word in tokens]
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        text=" ".join(tokens)
        return text
    
    def preprocess(self,dataset,remove_stopwords=True,lemmatize=True,stem=True):
        print("Starting complete data preprocessing")
        print("Executing tokenize method with args remove_stopwords ={} , lemmatize={} ,stem={} ".format(remove_stopwords,lemmatize,stem))
        dataset["transformed_text"] = dataset["transformed_text"].map(
            lambda x: self.tokenize(x,remove_stopwords,lemmatize,stem))
        print("Data preprocessing completed.")
        return dataset
