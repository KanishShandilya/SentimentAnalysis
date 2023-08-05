from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix




class TrainingModels:

    """
        Trains model based on given parameters

        Written By-Kanish Shandilya
    """

    def __init__(self,X_train,y_train):

        """
            X_train: Features to train model on
            y_train: labels
        """
        self.X=X_train
        self.y=y_train

    def trainLogisticRegression(self,penalty='l2',alpha=0.0001):
        """
            for penalty and alpha see SGDClassifier documentation 
        """
        print("Training logistic Regression with following parameters \n penalty={}\nalpha={}".format(penalty,alpha))

        lr=SGDClassifier(loss="log_loss",penalty=penalty,alpha=alpha)

        lr.fit(self.X,self.y)

        return lr
    
    def trainNaiveBayes(self):

        print("Training Naive Bayes classifier")

        nb=GaussianNB()

        nb.fit(self.X,self.y)

        return nb
    
    def evaluate(self,X_test,y_test,model):
        """
            X_test is test features data
            y_test is true labels for test data
        """
        print("Evaluating  model....")
        y_pred=model.predict(X_test)

        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred,average='micro')
        recall=recall_score(y_test,y_pred,average='micro')
        f1=f1_score(y_test,y_pred,average='micro')
        cm=confusion_matrix(y_test,y_pred)

        print("Accuracy is {}".format(accuracy))
        print("Precision is {}".format(precision))
        print("Recall is {}".format(recall))
        print("F1 score is {}".format(f1))

        print("Showing confusion matrix")
        print(cm)

        return accuracy
