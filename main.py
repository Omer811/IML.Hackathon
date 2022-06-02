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
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from preprocessing_noga import clean_cols
from Mission2_Breast_Cancer.Maya_features import preprocessing_by_maya, \
    hot_encoding_noga

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


def show_conf_matrix(y_true,y_pred,class_labels):

    for i in range (len(class_labels)):
        cm = confusion_matrix(y_true=y_true[:,i],  y_pred=y_pred[:,i],
                              labels=[0, 1])
        cm = ff.create_annotated_heatmap(cm, y=[r"$y = 0$", r"$y = 1$"],
                                         x=[r"$\widehat{y} = 0$",
                                            r"$\widehat{y}=1$"],
                                         annotation_text=np.core.defchararray.add(
                                             np.array([["TN: ", "FP: "],
                                                       ["FN: ", "TP: "]]),
                                             cm.astype("<U4")),
                                         showscale=True, colorscale="OrRd")
        cm.show()



def export_results(model:BaseEstimator, X_test,class_labels, y_test=None,
                   conf=False):
    X_test = transform_categorical(X_test)
    pred = model.predict(X_test)
    t_pred = transform_prediction_to_list(class_labels, pred)
    pd.Series(t_pred).to_csv("pred.csv",index=False)
    if y_test is not None:
        t_y_test = transform_prediction_to_list(class_labels, y_test)
        pd.Series(t_y_test).to_csv("y_test.csv",index=False)
    else:
        t_y_test = y_test
    if conf:
        show_conf_matrix(y_test,pred,class_labels)


def transform_prediction_to_list(class_labels, pred):
    class_labels = np.array(class_labels)
    pred = [[class_labels[i] if line[i] == 1 else None for i in range(len(
        line))] for line in pred.astype(int)]  # encode names of classes
    pred = [[class_name for class_name in line if class_name is not None]
            for line in
            pred]  # remove None values
    pred = [str(line) for line in pred]  # change to string
    return pred


if __name__ == '__main__':
    loader = Loader(path=X_PATH)
    loader.load()
    loader.activate_preprocessing([clean_cols, hot_encoding_noga,
                                   preprocessing_by_maya])
    loader.save_csv("pre_proc.csv")
    X=loader.get_data()
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

