from base_estimator import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class BaselineEstimatorMultipleClassifiers(BaseEstimator):
    def __init__(self,class_name:[str,...]):

        self.n_classes=len(class_name)
        self.classifiers = []
        for i in range(self.n_classes):
            self.classifiers.append(DecisionTreeClassifier())


    def _fit(self, X: pd.DataFrame, y: np.ndarray):
        for i in range(self.n_classes):
            self.classifiers[i].fit(X,y[:,i])

    def _predict(self, X: pd.DataFrame):
        pred = np.empty((X.shape[0],self.n_classes))
        for i in range(self.n_classes):
            pred[:,i] = self.classifiers[i].predict(X)
        return pred



    def _loss(self, X: pd.DataFrame, y: np.ndarray):
        y_pred = self.predict(X)
        return np.mean((y-y_pred)**2,axis=0)
