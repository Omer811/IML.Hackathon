import numpy as np
import pandas as pd
from data_loader import Loader


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
    df['אבחנה-M -metastases mark (TNM)'].fillna('Not yet Established', inplace=True)
    b = df['אבחנה-M -metastases mark (TNM)'].unique()
    df = pd.concat([df, pd.get_dummies(df['אבחנה-M -metastases mark (TNM)'])], axis=1)
    del df['אבחנה-M -metastases mark (TNM)']
    a = 1

    # deleting margin type as they are all marked exactly the same
    del df['אבחנה-Margin Type']


loader = Loader(
    path="C:\\Users\\Maya\\Desktop\\School\\IML\\hakathon\\IML.Hackathon\\Mission2_Breast_Cancer\\train.feats.csv")
loader.load()
df = loader.get_data()
preprocessing_by_maya(df)
a = 1
