import numpy as np
import pandas as pd
import sys
import plotly.graph_objects as go
from  data_loader import Loader

X_PATH = "Mission2_Breast_Cancer/train.feats.csv"


if __name__ == '__main__':
    loader = Loader(path=X_PATH)
    loader.load()
    df = loader.get_data().fillna(0).drop_duplicates()
    fig = go.Figure(data=go.Scatter(x=np.histogram(df[df.columns[-1]]),
                                    y=np.arange(len(df[df.columns[-1]]))))
    fig.show()
    #print (np.histogram(df[df.columns[-1]]))
    for col in df:
        print(f"______ COL_NAME: {col} ________")
        unique = df[col].unique()
        print(f"NUMBER OF UNIQUE ATTRS:{len(unique)}")
        print(df[col].unique())
