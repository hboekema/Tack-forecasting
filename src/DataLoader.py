
import os
import pandas as pd


class DataLoader:
    def __init__(self, datapath=None):
        self._set_working_directory()
        if datapath is None:
            self._set_datapath_if_unspecified()
        else:
            self.datapath = datapath
        
    def _set_working_directory(self):
        current_running_directory = os.getcwd()
        self.project_working_directory = current_running_directory.replace("notebooks", "").replace("src", "")

    def _set_datapath_if_unspecified(self):
        default_subdir_path = "data/test_data.csv"
        self.datapath = os.path.join(self.project_working_directory, default_subdir_path)
        
    def load_dataframe_from_datapath(self):
        indices_of_columns_to_parse_as_dates = [24]
        return pd.read_csv(self.datapath, parse_dates=indices_of_columns_to_parse_as_dates)
    
