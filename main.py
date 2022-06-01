import numpy as np
import pandas as pd
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.model_selection import train_test_split
from baseline_estimator import BaselineEstimator
from data_loader import Loader

X_PATH = "Mission2_Breast_Cancer/train.feats.csv"
Y_PATH = "Mission2_Breast_Cancer/train.labels.0.csv"

def preform_baseline(X_train: pd.DataFrame,y_train: pd.Series,
                     X_test: pd.DataFrame,y_test: pd.Series):
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns
    encoder = CountFrequencyEncoder(encoding_method='frequency',
                                    variables=categorical_features.to_list())
    encoder.fit(X_train)
    X_train = encoder.transform(X_train)

    bs = BaselineEstimator(class_name=["class1", "class2", "class3"])
    bs.fit(X_train, y_train)
    print(bs.loss(X_test,y_test))

def split_train_test_dev(X,y):
    X_temp,X_test, y_temp, y_test = train_test_split(X,y,test_size=0.6)
    X_train,X_dev, y_train, y_dev = train_test_split(X_temp,y_temp)

    return X_test,y_test,X_train, y_train,X_dev,y_dev
if __name__ == '__main__':
    loader = Loader(path=X_PATH)
    loader.load()
    X=loader.get_data().fillna(0).drop_duplicates()
    loader = Loader(path=Y_PATH)
    loader.load()
    y = loader.get_data().loc[X.index]

    X_test, y_test, X_train, y_train, X_dev, y_dev = split_train_test_dev(X,y)
    preform_baseline(X_train,y_train,X_dev,y_dev)

