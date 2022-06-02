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

    def _fit(self, X: pd.DataFrame, y: np.ndarray):
        self.u_estimator = OneVsRestClassifier(SVC(),
                                               n_jobs=self.n_jobs,
                                              verbose=49)
        # self.u_estimator = SVC()
        self.u_estimator.fit(X.to_numpy(), y)

    def _predict(self, X: pd.DataFrame):
        probs = self.u_estimator.predict(X.to_numpy())
        return probs


#pooled clustered ordinaly leaast squered

    def _loss(self, X: pd.DataFrame, y: np.ndarray):
        y_pred = self.predict(X)
        return np.mean(y-y_pred,axis=1)
