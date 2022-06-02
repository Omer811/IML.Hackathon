import numpy as np
import pandas as pd
from data_loader import Loader
import math
from preprocessing_noga import clean_cols
from Mission2_Breast_Cancer.Maya_features import preprocessing_by_maya, \
    hot_encoding_noga

loader = Loader(
    path="C:\\Users\\Maya\\Desktop\\School\\IML\\hakathon\\IML.Hackathon\\Mission2_Breast_Cancer\\train.feats.csv")
loader.load()
df = loader.get_data()
df = preprocessing_by_maya(df)
# Transform surgery dates values to datetime format, there are null values
df["אבחנה-Surgery date1"] = pd.to_datetime(df["אבחנה-Surgery date1"],
                                           errors='coerce')
df["אבחנה-Surgery date2"] = pd.to_datetime(df["אבחנה-Surgery date2"],
                                           errors='coerce')
df["אבחנה-Surgery date3"] = pd.to_datetime(df["אבחנה-Surgery date3"],
                                           errors='coerce')
df["surgery before or after-Activity date"] = pd.to_datetime(
    df["surgery before or after-Activity date"], errors='coerce')



df["אבחנה-Surgery date2"] = pd.to_datetime(df["אבחנה-Surgery date2"])
df["אבחנה-Surgery date1"] = pd.to_datetime(df["אבחנה-Surgery date1"])
df['time_between_1_2_surgery'] = df["אבחנה-Surgery date2"]-df["אבחנה-Surgery date1"]
df['time_between_1_2_surgery'] = df['time_between_1_2_surgery']/np.timedelta64(1,'D')


b = df['time_between_1_2_surgery']
c = b.dtype

a=1