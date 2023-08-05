import pickle
from train_models import TrainingModels





class Train:

    def __init__(self,X_train,y_train,X_test,y_test,read_json_obj):
        self.train_models=TrainingModels(X_train=X_train,y_train=y_train)
        self.X_test=X_test
        self.y_test=y_test
        self.read_json_obj=read_json_obj

    def train(self):
        try:
            #logistic Regression
            penalty=["l2"]
            alpha=[0.00001,0.0001]
            best_model=None
            best_acc=0
            for p in penalty:
                for a in alpha:
                    model=self.train_models.trainLogisticRegression(p,a)
                    accuracy=self.train_models.evaluate(self.X_test,self.y_test,model)

                    if accuracy > best_acc:
                        best_acc=accuracy
                        best_model=model

        
            logisticReg_file_path=self.read_json_obj.read_attribute("logisticReg_file_path")
            print("Saving the best logistic regression model at {}".format(logisticReg_file_path))
            self.save_model(best_model,logisticReg_file_path)
            #Naive Bayes
            nb_model=self.train_models.trainNaiveBayes()
            accuracy=self.train_models.evaluate(self.X_test,self.y_test,nb_model)

            nb_file_path=self.read_json_obj.read_attribute("nb_file_path")
            print("Saving the best Naive Bayes model at {}".format(nb_file_path))
        except Exception as e:
            print("Error occured at train method of training ",e)
    

    def save_model(self,model,file_path):
        """
            Saves given object in given file_path

            X: Object to be saved
            file_path: path to where file is to be saved
        """
        with open(file_path, "wb") as f:
            pickle.dump(model, f)


