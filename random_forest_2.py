from sklearn.ensemble import RandomForestClassifier
from base_estimator import BaseEstimator
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


class RandomForest2(BaseEstimator):
    def __init__(self):
        self.u_estimator = RandomForestClassifier()


    def _fit(self, X: pd.DataFrame, y: np.ndarray):
        self.u_estimator.fit(X.to_numpy(),y)

    def _predict(self, X: pd.DataFrame):
        return self.u_estimator.predict(X.to_numpy())

    def _loss(self, X: pd.DataFrame, y: np.ndarray):
        return mean_squared_error(self.predict(X),y)

