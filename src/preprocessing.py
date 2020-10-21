

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


def preprocess_dataframe_for_classification(dataframe):
    #feature_column_names = ["CurrentSpeed", "CurrentDir", "TWS", "TWA", "AWS", "AWA", "Roll", "Pitch", "HoG", "HeadingTrue", "SoG", "VMG", "RudderAng", "Leeway", "ModePilote"]
    feature_column_names = ["CurrentSpeed", "CurrentDir", "TWS", "TWA", "AWS", "AWA", "HoG", "HeadingTrue", "SoG", "VMG", "Leeway"]
    feature_dataframe = dataframe[feature_column_names]

    return feature_dataframe


def dataframe_to_dmatrix_for_classification(dataframe):
    preprocessed_dataframe = preprocess_dataframe_for_classification(dataframe)
    data_dmatrix = xgb.DMatrix(data=preprocessed_dataframe)

    return data_dmatrix


def preprocess_dataframe_for_classifier_training(dataframe):
    dataframe_labelled = dataframe.dropna(axis=0)
    X_dataframe = preprocess_dataframe_for_classification(dataframe_labelled)
    y_dataframe = dataframe_labelled["Tacking"]

    data_dmatrix = xgb.DMatrix(data=X_dataframe, label=y_dataframe)

    return data_dmatrix


def dataframe_to_dmatrix_for_classifier_training(dataframe):
    train_dataframe, val_dataframe = train_test_split(dataframe)

    train_dmatrix = preprocess_dataframe_for_classifier_training(train_dataframe)
    val_dmatrix = preprocess_dataframe_for_classifier_training(val_dataframe)

    return train_dmatrix, val_dmatrix


def order_dataframe_by_datetime(dataframe):
    # Boost performance by converting datetimes to seconds first
    seconds_since_earliest_sample = calculate_seconds_since_earliest_sample_in_dataframe(dataframe)
    sorted_datetimes = seconds_since_earliest_sample.sort_values()
    
    dataframe.index = sorted_datetimes.index
    sorted_dataframe = dataframe.sort_index()
    return sorted_dataframe


def calculate_seconds_since_earliest_sample_in_dataframe(dataframe):
    datetime_of_earliest_sample = dataframe["DateTime"].min()
    datetimes_since_earliest_sample = dataframe["DateTime"] - datetime_of_earliest_sample
    seconds_since_earliest_sample = datetimes_since_earliest_sample.dt.total_seconds()
    return seconds_since_earliest_sample


def datetime_to_seconds_since_earliest_sample_in_dataframe(dataframe):
    seconds_since_earliest_sample = calculate_seconds_since_earliest_sample_in_dataframe(dataframe)
    dataframe_seconds_since_earliest_sample = dataframe.assign(DateTime=seconds_since_earliest_sample)
    return dataframe_seconds_since_earliest_sample


def lag_array(array, front_value):
    cut_array = array[:-1]
    lagged_array = np.insert(cut_array, 0, front_value, axis=0)   # insert front_value at the front of cut_array
    return lagged_array


def calculate_diff_between_rows_in_series(series, front_value):
    # 'Lag' the series by one element and subtract from original series to efficiently calculate the time between consecutive samples
    array = series.to_numpy()
    lagged_array = lag_array(array, front_value)
    lagged_series = pd.Series(lagged_array)

    diff_between_samples = series - lagged_series
    return diff_between_samples


def calculate_seconds_between_samples_in_dataframe(dataframe):
    seconds_since_earliest_sample = calculate_seconds_since_earliest_sample_in_dataframe(dataframe)
    front_value = seconds_since_earliest_sample[0]

    seconds_between_samples = calculate_diff_between_rows_in_series(seconds_since_earliest_sample, front_value - 1)
    return seconds_between_samples


def split_dataframe_by_condition_mask(dataframe, split_mask):
    split_indices = np.where(split_mask)
    split_dataframes = np.split(dataframe, *split_indices, axis=0)
    return split_dataframes


def split_dataframe_into_time_sequences(dataframe, time_index=True):
    seconds_between_samples = calculate_seconds_between_samples_in_dataframe(dataframe)
    start_of_new_sequence = seconds_between_samples > 1
    time_sequences_dataframes = split_dataframe_by_condition_mask(dataframe, start_of_new_sequence)
    
    if time_index:
        for index, df in enumerate(time_sequences_dataframes):
            time_sequences_dataframes[index] = datetime_to_seconds_since_earliest_sample_in_dataframe(df.reset_index(drop=True))

    return time_sequences_dataframes


def split_time_sequence_by_tacking_label(time_sequence):
    tacking_series = time_sequence["Tacking"]
    front_value = tacking_series[0]
    diff_between_tack_rows = calculate_diff_between_rows_in_series(tacking_series, front_value)

    tack_started_or_stopped = diff_between_tack_rows != 0
    tack_event_sequences = split_dataframe_by_condition_mask(time_sequence, tack_started_or_stopped)

    return tack_event_sequences


def time_sequences_to_tack_segregated_sequences(time_sequences):
    tack_segregated_sequences = []
    for time_sequence in time_sequences:
        tack_event_sequences = split_time_sequence_by_tacking_label(time_sequence)
        tack_segregated_sequences.extend(tack_event_sequences)

    return tack_segregated_sequences


def preprocess_dataframe_for_forecasting(dataframe):
    feature_column_names= ["CurrentSpeed", "CurrentDir", "TWS", "TWA", "AWS", "AWA", "HoG", "HeadingTrue", "SoG", "VMG", "Leeway"]
    feature_dataframe = dataframe[feature_column_names]

    # Format data as time series
    seconds_since_earliest_sample = calculate_seconds_since_earliest_sample_in_dataframe(dataframe)
    feature_dataframe.index = seconds_since_earliest_sample

    return feature_dataframe


def preprocess_dataframe_for_forecasting_training(dataframe):
    dataframe_labelled = dataframe.dropna(axis=0)
    X_dataframe = preprocess_dataframe_for_forecasting(dataframe_labelled)
    
    return X_dataframe


