import pandas as pd
import sys


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1])
    for col in df:
        print(f"______ COL_NAME: {col} ________")
        unique = df[col].unique()
        print(f"NUMBER OF UNIQUE ATTRS:{len(unique)}")
        print(df[col].unique())
