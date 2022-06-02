import numpy as np
import pandas as pd
from data_loader import Loader
import math
from preprocessing_noga import clean_cols
from Mission2_Breast_Cancer.Maya_features import preprocessing_by_maya, \
    hot_encoding_noga

from preprocessing_noga import clean_cols
from Mission2_Breast_Cancer.Maya_features import preprocessing_by_maya, \
    hot_encoding_noga
from baseline_estimator_task2 import BaselineEstimatorRegression
from preprocessing_tomer import tomer_prep


def create_times(df):

    # turning diagnosis dates into time stamps
    df['אבחנה-Diagnosis date'] = pd.to_datetime(df['אבחנה-Diagnosis date'],
                                                errors='coerce')


    # time between 2 surgeries
    df['time_between_1_2_surgery'] = df["אבחנה-Surgery date2"] - df[
        "אבחנה-Surgery date1"]
    df['time_between_1_2_surgery'] = df[
                                         'time_between_1_2_surgery'] / np.timedelta64(
        1, 'D')
    df.loc[df['אבחנה-Surgery sum'] < 2, 'time_between_1_2_surgery'] = 0
    df['time_between_1_2_surgery'] = np.where(
        df['time_between_1_2_surgery'].astype(str) == 'nan', 0,
        df['time_between_1_2_surgery'])
    df['time_between_1_2_surgery'] = abs(df['time_between_1_2_surgery'])

    # time first surgery and diagnosis
    df['time_from_first_surgery'] = df['אבחנה-Diagnosis date'] - df[
        "אבחנה-Surgery date1"]
    df['time_from_first_surgery'] = df[
                                        'time_from_first_surgery'] / np.timedelta64(
        1, 'D')
    df.loc[df['אבחנה-Surgery sum'] < 1, 'time_from_first_surgery'] = 0
    df['time_from_first_surgery'] = np.where(
        df['time_from_first_surgery'].astype(str) == 'nan', 0,
        df['time_from_first_surgery'])
    df['time_from_first_surgery'] = abs(df['time_from_first_surgery'])

    del df["אבחנה-Surgery date2"]
    del df["אבחנה-Surgery date1"]
    del df["אבחנה-Surgery date3"]
    del df["surgery before or after-Activity date"]
    del df['אבחנה-Diagnosis date']

    return df


# main:
# loader = Loader(
#     path="C:\\Users\\Maya\\Desktop\\School\\IML\\hakathon\\IML.Hackathon\\Mission2_Breast_Cancer\\train.feats.csv")
# loader.load()
# df = loader.get_data()
# df = clean_cols(df)
# df = preprocessing_by_maya(df)
# df = create_times(df)
# b = df['אבחנה-Diagnosis date']
# a = 1
