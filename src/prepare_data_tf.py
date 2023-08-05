import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

class PrepareData:
    """
        Take preprocessed text and convert it to data that is ready for training

        data_path_train: InitialPreprocessed Data for training
        data_path_test: InitialPreprocessed Data for testing

        Written By-Kanish Shandilya
    """
    def __init__(self,data_path_train,data_path_test,read_jsonobj):
        self.data_path_train=data_path_train
        self.data_path_test=data_path_test
        self.read_jsonobj=read_jsonobj
        
    def prepareData(self,vocab_size=70000,max_length=80,trunc_type="post",oov_tok="<oov>"):
        """
            vocab_size: What is the vocabulary size
            max_length: max_length of sentences allowed
        """

        tokenizer_obj_path=self.read_jsonobj.read_attribute("tokenizer_obj_path")

        print("Reading train data from {}".format(self.data_path_train))
        train_df=pd.read_csv(self.data_path_train)
        print("Reading test data from {}".format(self.data_path_test))
        test_df=pd.read_csv(self.data_path_test)
        print("Coverting data to list")
        list_train=list(train_df["transformed_text"])
        list_test=list(test_df["transformed_text"])
        print("Tokenizing given sentences")
        tokenizer=Tokenizer(num_words=vocab_size,oov_token=oov_tok)
        tokenizer.fit_on_texts(list_train)
        print("Saving tokenizer objects")
        self.save_tokenizer(tokenizer,tokenizer_obj_path)
        train_seq=tokenizer.texts_to_sequences(list_train)
        train_padded_seq=pad_sequences(train_seq,maxlen=max_length,truncating=trunc_type)

        test_seq=tokenizer.texts_to_sequences(list_test)
        test_padded_seq=pad_sequences(test_seq,maxlen=max_length,truncating=trunc_type)
        print("Text transformation completed returning output")
        return (train_padded_seq,test_padded_seq)
    
    def load_tokenizer(self,path):
        """
            Load tokenizer from file path

            path:Where to load file from
        """
        with open(path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer
    

    def save_tokenizer(self,tokenizer,path):
        """
            Save tokenizer to given file path

            tokenizer: obj to save
            path: where to save
        """
        with open(path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

