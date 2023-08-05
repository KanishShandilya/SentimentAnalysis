from tf_model import Model
from prepare_data_tf import PrepareData
import os
import tensorflow as tf


class TrainTFModel:
    """
        Class to train tensorflow model

        Written By: Kanish Shandilya
    """
    def __init__(self,data_train,data_test,y_train,y_test,read_json_obj):
        """
            data_train: Where training data resides
            data_test:where testing data resides
            y_train_path:labels for training data
            y_test: labels for testing data
        """
        self.data_train=data_train
        self.data_test=data_test
        self.y_train=y_train
        self.y_test=y_test
        self.read_json_obj=read_json_obj

    def train(self):
        try:
            model_save_path=self.read_json_obj.read_attribute("model_save_path")
            print("getting model objects")

            model=Model(vocab_size=100000,embedding_dim=128,max_length=120,num_classes=3).get_model()

            print("Getting X_train and X_test")

            prepareData=PrepareData(self.data_train,self.data_train,self.read_json_obj)

            X_train,X_test=prepareData.prepareData(vocab_size=100000,max_length=120)

            

            print("Preparing model for training")

            model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

            print("Starting training")
            print(X_train.shape," ",self.y_train.shape)
            print(X_test.shape," ",self.y_test.shape)
            model.fit(X_train,self.y_train,epochs=15)

            print("Training finished!Saving model")

            model.save(model_save_path)

            print("Model saved")
            
        except Exception as e:
            print("Error occured at train function in train_tfmodel ",e)

        