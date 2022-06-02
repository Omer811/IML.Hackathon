from base_estimator import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np


class BaselineEstimator(BaseEstimator):
    def __init__(self,class_name:[str,...], n_jobs=4):
        self.n_jobs = n_jobs
        self.u_estimator = None
        self.n_classes=len(class_name)
        self.classes=class_name

    def _fit(self, X: pd.DataFrame, y: pd.Series):
        self.u_estimator = OneVsRestClassifier(SVC(),
                                               n_jobs=self.n_jobs)
        self.u_estimator.fit(
            X.to_numpy(), y.to_numpy())

    def _predict(self, X: pd.DataFrame):
        probs = self.u_estimator.predict(X.to_numpy())
        return pd.Series(np.argsort(probs, axis=1)[:,:3])



    def _loss(self, X: pd.DataFrame, y: pd.Series):
        y_pred = self.predict(X)
        return np.mean(y.to_numpy()-y_pred.to_numpy(),axis=1)
