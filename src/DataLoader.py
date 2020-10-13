

import pandas as pd


class DataLoader:
    def __init__(self, columns_with_dates=[24]):
        self.columns_with_dates = columns_with_dates

    def load_dataframe_from_csv_datapath(csv_datapath):
        return pd.read_csv(csv_datapath, parse_dates=self.columns_with_dates)


