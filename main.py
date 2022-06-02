import numpy as np
import pandas as pd
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.model_selection import train_test_split
from baseline_estimator import BaselineEstimator
from baseline_estimator_2 import BaselineEstimatorMultipleClassifiers,BaseEstimator
from data_loader import Loader
from sklearn import preprocessing
import ast
from functools import reduce

X_PATH = "Mission2_Breast_Cancer/train.feats.csv"
Y_PATH = "Mission2_Breast_Cancer/train.labels.0.csv"

def evaluate(estimator,X_train: pd.DataFrame, y_train: pd.Series,
             X_test: pd.DataFrame, y_test: pd.Series, labels):
    X_train = transform_categorical(X_train)
    X_test = transform_categorical(X_test)

    model = estimator(class_name=labels)
    model.fit(X_train, y_train)
    print(model.loss(X_test,y_test))
    return model

def transform_categorical(data:pd.DataFrame):
    categorical_features = data.select_dtypes(exclude=[np.number]).columns
    encoder = CountFrequencyEncoder(encoding_method='frequency',
                                    variables=categorical_features.to_list())
    encoder.fit(data)
    return encoder.transform(data)
def split_train_test_dev(X,y):

    X_temp,X_test, y_temp, y_test = train_test_split(X,y,test_size=0.9)
    X_train,X_dev, y_train, y_dev = train_test_split(X_temp,y_temp)

    return X_test,y_test,X_train, y_train,X_dev,y_dev

def get_unique_labels(y:pd.Series):
    unique = y.unique()
    unique = [ast.literal_eval(val) for val in unique]
    unique = list(reduce(lambda x,y:x+y,unique))
    return list(set(unique))

def binarize_y(y:pd.Series)->pd.Series:
    unique_lables = get_unique_labels(y)
    y = y.to_list()
    y = [ast.literal_eval(val) for val in y]

    lb = preprocessing.MultiLabelBinarizer(classes = unique_lables)
    #unique_labels = get_unique_labels(y)
    y_transformed = lb.fit_transform(y)
    #y = lb.transform(y)
    return y_transformed,unique_lables


def export_results(model:BaseEstimator, X_test,class_labels, y_test=None):
    X_test = transform_categorical(X_test)
    pred = model.predict(X_test)
    pred = transform_prediction_to_list(class_labels, pred)
    pd.Series(pred).to_csv("pred.csv",index=False)
    if y_test is not None:
        y_test = transform_prediction_to_list(class_labels, y_test)
        pd.Series(y_test).to_csv("y_test.csv",index=False)


def transform_prediction_to_list(class_labels, pred):
    class_labels = np.array(class_labels)
    pred = [[class_labels[i] if i == 1 else None for i in line] for line in \
            pred.astype(int)]  # encode names of classes
    pred = [[class_name for class_name in line if class_name is not None]
            for line in
            pred]  # remove None values
    pred = [str(line) for line in pred]  # change to string
    return pred


if __name__ == '__main__':
    loader = Loader(path=X_PATH)
    loader.load()
    X=loader.get_data().fillna(0).drop_duplicates()
    loader = Loader(path=Y_PATH)
    loader.load()
    y = loader.get_data().loc[X.index]
    y = y.squeeze()#transform to Series
    y,y_labels = binarize_y(y)

    X_test, y_test, X_train, y_train, X_dev, y_dev = split_train_test_dev(X,y)
    model = evaluate(BaselineEstimatorMultipleClassifiers,X_train, y_train,
                  X_dev,
             y_dev, y_labels)

    export_results(model,X_test,y_labels,y_test)

