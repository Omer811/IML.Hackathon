import numpy as np
import pandas as pd
from data_loader import Loader
import math


def cleanup_duplicates(df):
    #  remove duplicates fronm data





def preprocessing_by_maya(df):
    # df.rename(columns={' Form Name': 'Form Name'}, inplace=True)

    # Form Name - Replace one category with another, and divide into dummies
    df[" Form Name"] = df[' Form Name'].replace(['אנמנזה סיעודית קצרה'],
                                                'אנמנזה סיעודית')
    df = pd.concat([df, pd.get_dummies(df[" Form Name"])], axis=1)
    del df[' Form Name']

    # Here we should consider to seperate LI into a seperate column, beacause it is not necasseriley lower than L1 and L2.
    df['אבחנה-Lymphatic penetration'].replace(
        {"Null": 0, "L0 - No Evidence of invasion": 1,
         "LI - Evidence of invasion": 2,
         "L1 - Evidence of invasion of superficial Lym.": 3,
         "L2 - Evidence of invasion of depp Lym.": 4, }, inplace=True)

    # I turned Null into not yet established, and created dummy variables
    df['אבחנה-M -metastases mark (TNM)'].fillna('Not yet Established',
                                                inplace=True)
    df = pd.concat([df, pd.get_dummies(df['אבחנה-M -metastases mark (TNM)'])],
                   axis=1)
    del df['אבחנה-M -metastases mark (TNM)']

    # Margin type as dummy variables. Should we connect "ללא" u "נקיים"?
    df = pd.concat([df, pd.get_dummies(df['אבחנה-Margin Type'])],
                   axis=1)
    del df['אבחנה-Margin Type']

    # Lymph nodes mark as dummies
    df['אבחנה-N -lymph nodes mark (TNM)'].replace(
        {'#NAME?': "", 'NX': 'Not yet Established'}, inplace=True)
    df['אבחנה-N -lymph nodes mark (TNM)'].fillna('Not yet Established',
                                                 inplace=True)
    df = pd.concat([df, pd.get_dummies(df['אבחנה-N -lymph nodes mark (TNM)'])],
                   axis=1)
    del df['אבחנה-N -lymph nodes mark (TNM)']

    df['אבחנה-Nodes exam'].replace({np.nan: 0, "": 0, 'nan': 0}, inplace=True)
    df['אבחנה-Nodes exam'].fillna(0, inplace=True)
    idx = df['אבחנה-Nodes exam'].apply(math.isnan)
    df.loc[idx, 'אבחנה-Nodes exam'] = 0

    # Replace missing values with 0
    df['אבחנה-Positive nodes']
    idx = df['אבחנה-Positive nodes'].apply(math.isnan)
    df.loc[idx, 'אבחנה-Positive nodes'] = 0

    # Replace missing values with unknowns and create dummies
    df['אבחנה-Side'] = np.where(df['אבחנה-Side'].astype(str) == 'nan', 'side unknown',
                                df['אבחנה-Side'])
    df = pd.concat([df, pd.get_dummies(df['אבחנה-Side'])], axis=1)
    del df['אבחנה-Side']

    dict = {'Stage0': 0, 'Stage0a': 0, 'Stage0is': 0, 'Stage1': 1, 'Stage1a': 1.3,
         'Stage1b': 1.5, 'Stage1c': 1.7, 'Stage2': 2, 'Stage2a': 2.3,
         'Stage2b': 2.6, 'Stage3': 3, 'Stage3a': 3.2, 'Stage3b': 3.5,
         'Stage3c': 3.8, 'Stage4': 4, 'LA': 3, 'nan':0, 'Not yet Established':0}
    for bad,good in dict.items():
        df['אבחנה-Stage'] = np.where(df['אבחנה-Stage'].astype(str) == bad, good, df['אבחנה-Stage'])



loader = Loader(
    path="C:\\Users\\Maya\\Desktop\\School\\IML\\hakathon\\IML.Hackathon\\Mission2_Breast_Cancer\\train.feats.csv")
loader.load()
df = loader.get_data()
preprocessing_by_maya(df)
a = 1
