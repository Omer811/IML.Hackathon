import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder


## אבחנה-Ivi -Lymphovascular invasion
def lymph_invasion(df):
    df["אבחנה-Ivi -Lymphovascular invasion"] = df["אבחנה-Ivi -Lymphovascular invasion"].map({
        "nan": 0, "none": 0, "+": 1, "extensive": 1, "-": 0,
        '': 0 , "No":0, "(-)": 0, "NO": 0, "(+)": 1, "neg": 0,
        "not": 0, "pos": 0, "yes": 1, "no": 0, None: 0,
        "MICROPAPILLARY VARIANT": 2})
    return df


## אבחנה-KI67 protein
def ki67(df):
    non_null = df.loc[~(df["אבחנה-KI67 protein"].isnull())]
    num_only = non_null.loc[non_null["אבחנה-KI67 protein"].str.match("^\d+(?:\.\d+)?$")]
    num_only["אבחנה-KI67 protein"] = (num_only["אבחנה-KI67 protein"].astype(float)) / 100

    percentage_num = non_null.loc[non_null["אבחנה-KI67 protein"].str.match("^\d+(?:\.\d+)?%$")]
    percentage_num["אבחנה-KI67 protein"] = (percentage_num["אבחנה-KI67 protein"].str.replace("%", "").astype(
        float)) / 100

    all_changes = pd.concat([num_only, percentage_num])
    df.update(all_changes)

    split_numbers = non_null["אבחנה-KI67 protein"].str.extract("^(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)%$").fillna(0).astype(
        int)
    mean_column = ((split_numbers[0] + split_numbers[1]) / 2) / 100
    mean_column.loc[mean_column == 0] = None

    for index, row in df.iterrows():
        if (index in mean_column.index) and (~np.isnan(mean_column[index])):
            df.loc[index, "אבחנה-KI67 protein"] = mean_column[index]

    return df


## 'אבחנה-Her2'
def her2(df):
    POSITIVE_REGEX = r"(pos|Pos|POS|po|\+|[0-9]+.*[0-9]*\%*|jhuch|חיובי|strong|Strong|STRONG|high|High|HIGH|beg)"
    NEGATIVE_REGEX = r"(neg|ned|Neg|NEG|ned|nge|akhah|\-|שלילי)"

    def parse_pos_neg_col(df, col_name):
        df[col_name] = np.where(df[col_name].str.contains(POSITIVE_REGEX), "1", df[col_name])
        df[col_name] = np.where(df[col_name].str.contains(NEGATIVE_REGEX), "0", df[col_name])

    parse_pos_neg_col(df, 'אבחנה-Her2')

    return df


# to convert to one-hot encoding
def one_hot(df):
    df.drop(df.index[df['אבחנה-Basic stage'] == 'Null'], inplace=True)
    encoder = OneHotEncoder(sparse=False)
    df_encoded = pd.DataFrame(
        encoder.fit_transform(df[[' Hospital', 'אבחנה-Basic stage', 'אבחנה-Histological diagnosis']]))
    df_encoded.columns = encoder.get_feature_names([' Hospital', 'אבחנה-Basic stage', 'אבחנה-Histological diagnosis'])
    df.drop([' Hospital', 'אבחנה-Basic stage', 'אבחנה-Histological diagnosis'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df_encoded.reset_index(drop=True, inplace=True)
    df = pd.concat([df, df_encoded], axis=1)

    return df


# to convert to ordinal encoding
def ordinal_enc(df):
    cat = ['G1 - Well Differentiated', 'G2 - Modereately well differentiated', 'G3 - Poorly differentiated',
           'G4 - Undifferentiated', 'GX - Grade cannot be assessed', 'Null']
    ordinal_encoder = OrdinalEncoder(categories=[cat])
    df['אבחנה-Histopatological degree'] = ordinal_encoder.fit_transform(df[['אבחנה-Histopatological degree']])

    ordinal_encoder = OrdinalEncoder()
    df['User Name'] = ordinal_encoder.fit_transform(df[['User Name']])

    return df


# fill nan, drop entries, convert values to float
def finishes(df):
    df.loc[df["אבחנה-KI67 protein"].isnull(), "אבחנה-KI67 protein"] = 0

    df = df.drop(df[~df['אבחנה-Her2'].isin([1, 0, '0', '1'])].index)
    df = df[pd.to_numeric(df["אבחנה-KI67 protein"], errors='coerce').notnull()]
    df["אבחנה-Ivi -Lymphovascular invasion"] = df["אבחנה-Ivi -Lymphovascular invasion"].astype(float)
    df['אבחנה-Her2'] = df['אבחנה-Her2'].astype(float)
    df["אבחנה-KI67 protein"] = df["אבחנה-KI67 protein"].astype(float)

    return df

def tomer_prep(df):
    df = lymph_invasion(df)
    df = ki67(df)
    df = her2(df)
    df = one_hot(df)
    df = ordinal_enc(df)
    df = finishes(df)
    return df
