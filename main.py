import numpy as np
import pandas as pd
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.model_selection import train_test_split
from baseline_estimator_2 import BaselineEstimatorMultipleClassifiers,BaseEstimator
from data_loader import Loader
from sklearn import preprocessing
import ast
from functools import reduce
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
from preprocessing_noga import clean_cols
from Mission2_Breast_Cancer.Maya_features import preprocessing_by_maya, \
    hot_encoding_noga
from feature_creation import create_times
from xgboost_2 import XGB2

X_PATH = "Mission2_Breast_Cancer/train.feats.csv"
X_PATH_PICKLED = "Mission2_Breast_Cancer/train.feats.csv.pickled"
Y_PATH_0 = "Mission2_Breast_Cancer/train.labels.0.csv"
Y_PATH_1 = "Mission2_Breast_Cancer/train.labels.1.csv"

def mean_ids(df:pd.DataFrame,y0,y1):
    df = pd.concat([df,y0,y1], axis=1)
    df = df.dropna()

    df = df.groupby('id-hushed_internalpatientid', as_index=False).mean()

    y0 = df[df.columns[-12:-1]]
    y1 = df[df.columns[-1]]
    df= df.drop(columns=df.columns[-12:])
    return df,y0,y1

def drop_dates(df:pd.DataFrame):
    ids = df["id-hushed_internalpatientid"]
    dates = df.select_dtypes(exclude=[np.number]).columns
    df = df.drop(columns=dates)
    df["id-hushed_internalpatientid"] = ids
    return df

def evaluate_1(estimator,X_train: pd.DataFrame, y_train: pd.Series,
             X_test: pd.DataFrame, labels):
    X_train = transform_categorical(X_train)

    model = estimator(labels)
    # model = estimator()
    model.fit(X_train, y_train.to_numpy())
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
    X = X.sample(frac=1)
    unique_idx = X["id-hushed_internalpatientid"].unique()
    idx_split = np.array_split(unique_idx,2)
    X_test = X[X["id-hushed_internalpatientid"].isin(idx_split[0])]
    train_X = X[X["id-hushed_internalpatientid"].isin(idx_split[1])]
    del X_test["id-hushed_internalpatientid"]
    del train_X["id-hushed_internalpatientid"]
    y_test = y.loc[X_test.index]
    y_train = y.loc[train_X.index]
    X_train, X_dev, y_train, y_dev = train_test_split(train_X,y_train,
                                                   test_size=0.7)


    return X_test,y_test.to_numpy(),X_train, y_train.to_numpy(),X_dev,y_dev.to_numpy()

def get_unique_labels(y:pd.Series):
    unique = y.unique()
    unique = [ast.literal_eval(val) for val in unique]
    unique = list(reduce(lambda x,y:x+y,unique))
    return list(set(unique))

def binarize_y(y:pd.DataFrame)->pd.Series:
    idx = y.index
    unique_lables = get_unique_labels(y)
    y = y.to_list()
    y = [ast.literal_eval(val) for val in y]

    lb = preprocessing.MultiLabelBinarizer(classes = unique_lables)
    #unique_labels = get_unique_labels(y_0)
    y_transformed = pd.DataFrame(lb.fit_transform(y))
    #y_0 = lb.transform(y_0)
    y_transformed.set_index(idx)
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
             X_test: pd.DataFrame):
    X_train = transform_categorical(X_train)
    X_test = transform_categorical(X_test)

    model = estimator()
    model.fit(X_train, y_train)

    return model

def submit():
    loader = Loader(path=X_PATH, pickled_path=X_PATH_PICKLED)
    loader.load()
    loader.activate_preprocessing([clean_cols, hot_encoding_noga,
                                   preprocessing_by_maya, create_times,
                                   drop_dates])
    X = loader.get_data()
    loader = Loader(path=Y_PATH_0)
    loader.load()
    y_0 = loader.get_data().loc[X.index]
    y_0_name = y_0.columns[0]
    y_0 = y_0.squeeze().rename(y_0_name)  # transform to Series

    y_0, y_labels = binarize_y(y_0)
    loader = Loader(path=Y_PATH_1)
    loader.load()
    y_1 = loader.get_data().loc[X.index]
    y_1_name = y_1.columns[0]
    y_1 = y_1.squeeze().rename(y_1_name)  # transform to Series

    X, y_0, y_1 = mean_ids(X, y_0, y_1)
    y_1.rename(y_1_name)

    loader = Loader(path="Mission2_Breast_Cancer/test.feats.csv")
    loader.load()
    loader.activate_preprocessing([clean_cols, hot_encoding_noga,
                                  preprocessing_by_maya,create_times,
                                     drop_dates])
    X_test = loader.get_data()
    del X["id-hushed_internalpatientid"]
    del X_test["id-hushed_internalpatientid"]
    X, X_test = X.align(X_test, join='outer', axis=1, fill_value=0)
    model = evaluate_1(BaselineEstimatorMultipleClassifiers, X, y_0,
                       X_test,labels= y_labels)

    export_results(model, "task1_", X_test, y_labels, None, y_name=y_0_name,
                   conf=False)


    model = evaluate_2(XGB2, X, y_1, X_test)

    export_results(model, "task2_", X_test, y_labels, None, y_name=y_1_name,
                   parse_y=False)

if __name__ == '__main__':
    np.random.seed(0)
    submit()
    exit()
    loader = Loader(path=X_PATH,pickled_path=X_PATH_PICKLED)
    loader.load()
    loader.activate_preprocessing([clean_cols, hot_encoding_noga,
                                   preprocessing_by_maya,create_times,
                                   drop_dates])
    # loader.save_csv("pre_proc.csv")
    # loader.pickle_data()
    # loader.load_pickled()
    X = loader.get_data()
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

    X,y_0,y_1 = mean_ids(X,y_0,y_1)
    y_1.rename(y_1_name)


    X_test, y_test, X_train, y_train, X_dev, y_dev = split_train_test_dev(X,
                                                                          y_0)
    model = evaluate_1(BaselineEstimatorMultipleClassifiers,X_train, y_train,
                  X_dev,  y_dev, y_labels)

    export_results(model,"task1_",X_test,y_labels,y_test,y_name=y_0_name,
                   conf=False)
    X_test, y_test, X_train, y_train, X_dev, y_dev = split_train_test_dev(X,
                                                                          y_1)
    model = evaluate_2(XGB2, X_train, y_train,
                       X_dev, y_dev)

    export_results(model, "task2_", X_test, y_labels, y_test,y_name=y_1_name,
                   parse_y=False)


