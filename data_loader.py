import pandas as pd


class Loader:
    def __init__(self, path: str = "", pickled_path: str = ""):
        self.path = path
        self.pickled_path = ""
        self.df: pd.DataFrame = None

    def load(self):
        self.df = pd.read_csv(self.path, parse_dates=True)

    def activate_preprocessing(self, pre_processing_functions):
        for fun in pre_processing_functions:
            self.df = fun(self.df)

    def pickle_data(self):
        if self.df is not None:
            self.df.to_pickle(self.path + ".pickled")
            self.pickled_path = self.path + ".pickled"

    def load_pickled(self):
        self.df = pd.read_pickle(self.pickled_path)

    def get_data(self) -> pd.DataFrame:
        if self.df is not None:
            return self.df
