


import numpy as np
import matplotlib.pyplot as plt

from DataLoader import DataLoader
from TackClassifier import TackClassifier
from TackForecaster import TackForecaster

from preprocessing import split_dataframe_into_time_sequences


import argparse

parser = argparse.ArgumentParser(description="Evaluate tack forecasting model on the data provided")
parser.add_argument("--datapath", type=str, help="path to CSV dataset to be used for evaluation")

args = parser.parse_args()


if __name__ == "__main__":
    if args.datapath is not None:
        time_sequence_for_eval = DataLoader(args.datapath).load_dataframe_from_datapath() 
    else:
        # Load dataset
        dataframe = DataLoader("../data/test/test_data.csv").load_dataframe_from_datapath()
        time_sequences = split_dataframe_into_time_sequences(dataframe, time_index=False)
        
        index_to_use = np.random.randint(0, len(time_sequences) - 1)
        time_sequence_for_eval = time_sequences[index_to_use]
        #print(time_sequence_for_eval)
        #print(time_sequence_for_eval.dtypes)
    time_sequence_for_eval_length = len(time_sequence_for_eval.index)
    print("Time sequence length: %d" % time_sequence_for_eval_length)

    # Set up forecasting model
    classifier_model = TackClassifier()
    classifier_model.load_model("../models/xgb_classifier.model")
   
    max_lag_order = 5
    forecast_window_in_seconds = 1
    forecaster_model = TackForecaster(classifier_model, max_lag_order, forecast_window_in_seconds)

    predicted_labels = []
    true_labels = []

    for index in range(len(time_sequence_for_eval.index)):
        row_as_series = time_sequence_for_eval.iloc[index]
        row = row_as_series.to_frame().T
        row = row.convert_dtypes(convert_integer=False)
        
        tacking = forecaster_model.forecast_from_datapoint(row)
        #print("Predicted tacking label: %d" % tacking)
        if index + forecast_window_in_seconds < time_sequence_for_eval_length:
            #true_tacking_label = dataframe["Tacking"].iloc[index + forecast_window_in_seconds]
            true_tacking_label = dataframe["Tacking"].iloc[index]
            #print("True tacking label: %d\n" % true_tacking_label)
            if tacking != -1:
                predicted_labels.append(tacking)
                true_labels.append(int(true_tacking_label))
        else:
            pass
            #print("No GT tacking label\n")
        
        if len(predicted_labels) > 0:
            true_preds = ~np.logical_xor(predicted_labels, true_labels)
            accuracy = float(true_preds.sum()) / float(len(true_preds))
            print("Accuracy: %.03f" %  accuracy)

    #print(predicted_labels)
    #print(true_labels)

    true_preds = ~np.logical_xor(predicted_labels, true_labels)
    accuracy = float(true_preds.sum()) / float(len(true_preds))
    print("Accuracy: %.03f" %  accuracy)

    time_steps = range(len(true_labels))
    plt.plot(time_steps, true_labels, label="true labels")
    plt.plot(time_steps, predicted_labels, label="predicted labels")
    plt.show()


