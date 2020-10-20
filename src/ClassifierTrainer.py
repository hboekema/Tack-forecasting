


from sklearn.model_selection import KFold, train_test_split
from TackClassifier import TackClassifier


class ClassifierTrainer:
    def __init__(self, classifier):
        self.classifier = classifier

    def fit_with_kfold_validation(self, data_dmatrix, **kwargs):
        print("Training with K-fold cross-validation.")
        self.classifier.cv(data_dmatrix, **kwargs)
        

    def fit_with_validation_set(self, data_dmatrix, **kwargs):
        print("Training with validation set.")
        self.classifier.train(data_dmatrix, **kwargs)


    def fit(self, data_dmatrix, **kwargs):
        if 'nfold' in kwargs:
            self.fit_with_kfold_validation(data_dmatrix, **kwargs)
        else:
            self.fit_with_validation_set(data_dmatrix, **kwargs)
        
        return self.classifier 
