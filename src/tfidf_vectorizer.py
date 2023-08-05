from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle



class TFIDF_Vectorizer:

    def __init__(self,dataPath,ngram_range=(1,1),max_df=250000,min_df=40,max_features=70000,norm='l2',read_json_obj=None):
        """
            dataPath: Path of completePreprocessed Data
            (ngram_range,max_df,min_df,max_features,norm)->TfidfVectorizer parameters

            Written By: Kanish Shandilya
        """
        self.dataPath=dataPath
        self.ngram_range=ngram_range
        self.max_df=max_df
        self.min_df=min_df
        self.max_features=max_features
        self.norm=norm
        self.read_json_obj=read_json_obj

    def get_features(self):
        """
            returns featurized array of text
        """
        obj_save=self.read_json_obj.read_attribute("tfidf_obj_save")
        print("Reading the data from {}".format(self.dataPath))

        df=pd.read_csv(self.dataPath)

        print("Read the data. Now creating TFIdf object and fitting data in it")

        tfIdf=TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_df=self.max_df,
            min_df=self.min_df,
            max_features=self.max_features,
            norm=self.norm
        )

        X=tfIdf.fit_transform(df["transformed_text"])

        print("Done fit and transform data. Now saving object in file")

        self.save_object(tfIdf, obj_save)

        print("Saved object in File. Now return X after converting it to array")

        X=X.toarray()

        return X
    
    def save_object(self,X,file_path):
        """
            Saves given object in given file_path

            X: Object to be saved
            file_path: path to where file is to be saved
        """
        with open(file_path, "wb") as f:
            pickle.dump(X, f)

    def transform(self,file_path,complete_preprocess_test_path):
        """
            Reads object from given file_path

            file_path: path from where file is to be read
        """

        print("Reading the data from {}".format(complete_preprocess_test_path))

        df=pd.read_csv(complete_preprocess_test_path)
        with open(file_path, "rb") as f:
            X = pickle.load(f)
        return X.transform(df["transformed_text"])
