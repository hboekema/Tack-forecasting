

import pandas as pd
import numpy as np

from DataLoader import DataLoader
from preprocessing import split_dataframe_into_time_sequences


def split_train_test(csv_datapath=None):
    dataframe = DataLoader(csv_datapath).load_dataframe_from_datapath()
    
    # Drop rows with values missing
    dataframe = dataframe.dropna().reset_index(drop=True)

    time_sequences = split_dataframe_into_time_sequences(dataframe, time_index=False)
    print(time_sequences[0])


    # There are 179 complete sequences, so choose 20% (36) of these at random and store as test data
    test_indices = np.random.choice(len(time_sequences), round(0.2*len(time_sequences)), replace=False)
    print(test_indices)
    train_indices = [i for i in range(len(time_sequences)) if i not in test_indices]

    test_sequences = [time_sequences[i] for i in test_indices]
    train_sequences = [time_sequences[i] for i in train_indices]

    test_dataframe = pd.concat(test_sequences)
    train_dataframe = pd.concat(train_sequences)

    print(test_dataframe)

    test_dataframe.to_csv("../data/test/test_data.csv")
    train_dataframe.to_csv("../data/train/train_data.csv")


if __name__ == "__main__":
    split_train_test("../data/test_data.csv")


