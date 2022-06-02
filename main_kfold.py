import numpy as np
import pandas as pd
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.model_selection import train_test_split, KFold
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
from feature_creation import create_times
from baseline_estimator_task2 import BaselineEstimatorRegression
from preprocessing_tomer import tomer_prep
import evaluate_0_pycharm
import evaluate_1_pycharm

X_PATH = "Mission2_Breast_Cancer/train.feats.csv"
X_PATH_PICKLED = "Mission2_Breast_Cancer/train.feats.csv.pickled"
Y_PATH_0 = "Mission2_Breast_Cancer/train.labels.0.csv"
Y_PATH_1 = "Mission2_Breast_Cancer/train.labels.1.csv"

def mean_ids(df:pd.DataFrame):
    df = pd.concat([df,df['id-hushed_internalpatientid']], axis=1)
    df = df.groupby('id-hushed_internalpatientid', as_index=False).mean()
    df.drop(["id-hushed_internalpatientid"], axis=1, inplace=True)
    return df

def drop_dates(df:pd.DataFrame):
    dates = df.select_dtypes(exclude=[np.number]).columns
    df = df.drop(columns=dates)
    return df

def evaluate_1(estimator,X_train: pd.DataFrame, y_train: pd.Series,
             X_test: pd.DataFrame, y_test: pd.Series, labels):

    X_train = transform_categorical(X_train)
    X_train.to_csv("for_maya_she_doesnt_believe_in_computers.csv")
    X_test = transform_categorical(X_test)

    model = estimator(labels)
    model.fit(X_train, y_train)
    print(model.loss(X_test,y_test))
    return model

def transform_categorical(data:pd.DataFrame):
    categorical_features = data.select_dtypes(exclude=[np.number]).columns

    if len(categorical_features)>0:
        encoder = CountFrequencyEncoder(encoding_method='frequency',
                                        variables=categorical_features.to_list())
        encoder.fit(data)
        return encoder.transform(data)
    return data
def split_train_test_dev(X,y):

    X_temp,X_test, y_temp, y_test = train_test_split(X,y,test_size=0.7)
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
    #unique_labels = get_unique_labels(y_0)
    y_transformed = lb.fit_transform(y)
    #y_0 = lb.transform(y_0)
    return y_transformed,unique_lables


def show_conf_matrix(y_true,y_pred,class_labels):

    for i in range (len(class_labels)):
        cm = confusion_matrix(y_true=y_true[:,i],  y_pred=y_pred[:,i],
                              labels=[0, 1])
        cm = ff.create_annotated_heatmap(cm, y=[r"$y_0 = 0$", r"$y_0 = 1$"],
                                         x=[r"$\widehat{y_0} = 0$",
                                            r"$\widehat{y_0}=1$"],
                                         annotation_text=np.core.defchararray.add(
                                             np.array([["TN: ", "FP: "],
                                                       ["FN: ", "TP: "]]),
                                             cm.astype("<U4")),
                                         showscale=True, colorscale="OrRd")
        cm.update_layout({"title":class_labels[i]})
        cm.show()



def export_results(model:BaseEstimator,path, X_test,class_labels,
                   y_test:pd.Series,y_name,
                   conf=False,parse_y = True):
    X_test = transform_categorical(X_test)
    pred = model.predict(X_test)
    t_pred = pred
    if parse_y:
        t_pred = transform_prediction_to_list(class_labels, pred)
    pd.Series(t_pred,name = y_name).to_csv(path+"y_pred.csv",index=False)
    t_y_test = y_test
    if y_test is not None:
        if parse_y:
            t_y_test = transform_prediction_to_list(class_labels, y_test)
        pd.Series(t_y_test,name = y_name).to_csv(path+"y_test.csv",index=False)

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

def evaluate_2(estimator,X_train: pd.DataFrame, y_train: pd.Series,
             X_test: pd.DataFrame, y_test: pd.Series):
    X_train = transform_categorical(X_train)
    X_test = transform_categorical(X_test)

    model = estimator()
    model.fit(X_train, y_train)
    print(model.loss(X_test, y_test))
    return model

def normal_run(X, y_0, y_1):
    X_test, y_test, X_train, y_train, X_dev, y_dev = split_train_test_dev(X, y_0)
    model = evaluate_1(BaselineEstimatorMultipleClassifiers, X_train, y_train,
                       X_dev,
                       y_dev, y_labels)

    export_results(model, "task1_", X_test, y_labels, y_test, y_name=y_0_name,
                   conf=False)
    X_test, y_test, X_train, y_train, X_dev, y_dev = split_train_test_dev(X,
                                                                          y_1)
    model = evaluate_2(BaselineEstimatorRegression, X_train, y_train,
                       X_dev, y_dev)

    export_results(model, "task2_", X_test, y_labels, y_test, y_name=y_1_name,
                   parse_y=False)

def kfold_run(X, y_0, y_labels, y_1, k=5):

    kfold = KFold(n_splits=k, shuffle=True, random_state=1)

    micro_f1_list = []
    macro_f1_list = []
    for train, test in kfold.split(X, y_0):
        X_train, y_train = X.iloc[train], y_0[train]
        X_test, y_test = X.iloc[test], y_0[test]
        model = evaluate_1(BaselineEstimatorMultipleClassifiers, X_train, y_train,X_test,y_test, y_labels)
        export_results(model, "task1_", X_test, y_labels, y_test, y_name=y_0_name,conf=False)
        micro_f1, macro_f1 = evaluate_0_pycharm.run("task1_y_test.csv", "task1_y_pred.csv")
        micro_f1_list.append(micro_f1)
        macro_f1_list.append(macro_f1)

    print("Micro f1:" + str(np.mean(micro_f1_list)))
    print("Macro f1:" + str(np.mean(macro_f1_list)))

    mse_list = []
    for train, test in kfold.split(X, y_1):
        X_train, y_train = X.iloc[train], y_1[train]
        X_test, y_test = X.iloc[test], y_1[test]
        model = evaluate_2(BaselineEstimatorRegression, X_train, y_train,X_test,y_test)
        export_results(model, "task2_", X_test, y_labels, y_test, y_name=y_1_name,parse_y=False)
        mse = evaluate_1_pycharm.run("task2_y_test.csv", "task2_y_pred.csv")
        mse_list.append(mse)

    print("MSE:" + str(np.mean(mse_list)))


if __name__ == '__main__':

    loader = Loader(path=X_PATH,pickled_path=X_PATH_PICKLED)
    loader.load()
    # loader.activate_preprocessing([clean_cols, hot_encoding_noga,
    #                                preprocessing_by_maya,
    #                                tomer_prep,create_times,mean_ids,
    #                                drop_dates])
    loader.activate_preprocessing([clean_cols])
    loader.save_csv("pre_proc.csv")
    loader.pickle_data()
    # loader.load_pickled()
    X = loader.get_data().fillna(0)
    loader = Loader(path=Y_PATH_0)
    loader.load()
    y_0 = loader.get_data().loc[X.index]
    y_0_name = y_0.columns[0]
    y_0 = y_0.squeeze().rename(y_0_name)#transform to Series

    y_0, y_labels = binarize_y(y_0)
    loader = Loader(path=Y_PATH_1)
    loader.load()
    y_1 = loader.get_data().loc[X.index]
    y_1_name = y_1.columns[0]
    y_1 = y_1.squeeze().rename(y_1_name) # transform to Series

    #normal_run(X,y_0, y_1)
    kfold_run(X, y_0, y_labels, y_1)

