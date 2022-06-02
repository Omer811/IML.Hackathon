import numpy as np
import pandas as pd
from collections import defaultdict


TNM_MAP = r"Mission2_Breast_Cancer/tnm.txt"
NAN = 0
OTHER = "other"
POSITIVE_REGEX = r"(pos|Pos|POS|po|\+|[0-9]+.*[0-9]*\%*|jhuch|חיובי|strong|Strong|STRONG|high|High|HIGH|beg)"
NEGATIVE_REGEX = r"(neg|ned|Neg|NEG|ned|nge|akhah|\-|שלילי)"


#Default value
def def_value():
    return "Undefined"

#Default value
def def_value_zero():
    return 0

# Get file in format {input}:{tag}, return dictionary
def load_parse_file(file):
    parse_values = defaultdict(def_value)
    with open(file) as f:
        for line in f:
            if line:
                (key, val) = line.split(",")
                parse_values[key] = val[:-1]
    return parse_values


# Parse column and normalize pos\neg values -> Positive,Negative,Unknown
def parse_pos_neg_col(df, col_name):
    df[col_name] = np.where(df[col_name].str.contains(POSITIVE_REGEX), "Positive", "Negative")
    # df[col_name] = np.where(df[col_name].str.contains(NEGATIVE_REGEX), "Negative", df[col_name])

# Map each column according to dict values
def map_col(col, file):
    parse_values = load_parse_file(file)
    return col.map(parse_values)


def clean_cols(df):
    # Tumor depth > 0, replace negative and null with 0
    df["אבחנה-Tumor depth"].fillna(0, inplace=True)
    df.loc[df["אבחנה-Tumor depth"] < 0] = 0

    # Tumor width > 0, replace null values with 0
    df["אבחנה-Tumor width"].fillna(0, inplace=True)
    df.loc[df["אבחנה-Tumor width"] < 0] = 0

    # Replace weird input values with pre-tagged values
    parse_pos_neg_col(df, "אבחנה-er")
    parse_pos_neg_col(df, "אבחנה-pr")
    df["אבחנה-T -Tumor mark (TNM)"] = map_col(df["אבחנה-T -Tumor mark (TNM)"], TNM_MAP)

    # Add feture Tumor-Size
    df["tumor_size"] = df["אבחנה-T -Tumor mark (TNM)"]
    t_map = defaultdict(def_value_zero)
    t_map["T1"], t_map["T2"], t_map["T3"], t_map["T4"] = 1, 2, 3, 4
    df["tumor_size"] = df["tumor_size"].map(t_map)

    # Replace null surgery values with "other"
    df["surgery before or after-Actual activity"].fillna(OTHER, inplace=True)  # 10 values
    df["אבחנה-Surgery name1"].fillna(OTHER, inplace=True)  # 23 values
    df["אבחנה-Surgery name2"].fillna(OTHER, inplace=True)  # 18 values
    df["אבחנה-Surgery name3"].fillna(OTHER, inplace=True)  # 6 values

    # Transform surgery dates values to datetime format, there are null values
    df["אבחנה-Surgery date1"] = pd.to_datetime(df["אבחנה-Surgery date1"], errors='coerce')
    df["אבחנה-Surgery date2"] = pd.to_datetime(df["אבחנה-Surgery date2"], errors='coerce')
    df["אבחנה-Surgery date3"] = pd.to_datetime(df["אבחנה-Surgery date3"], errors='coerce')
    df["surgery before or after-Activity date"] = pd.to_datetime(df["surgery before or after-Activity date"],
                                                                 errors='coerce')

    # Drop id column
    df.drop(["id-hushed_internalpatientid"], axis=1, inplace=True)

    # Surgery sum > 0, replace negative and null with 0
    df["אבחנה-Surgery sum"].fillna(0, inplace=True)

    return df
