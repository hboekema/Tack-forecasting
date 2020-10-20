


from TackClassifier import TackClassifier
from TackForecaster import TackForecaster

import argparse

parser = argparse.ArgumentParser(description="Run tack forecasting model on the dataset provided. Assumes dataset is a single, regularly sample, ordered time sequence")
parser.add_argument("dataset", type=str, help="path to CSV dataset to be used for prediction")

args = parser.parse_args()


if __name__ == "__main__":
    # Load dataset
    dataframe = DataLoader(args["dataset"]).load_dataframe_from_datapath()
    dataframe_num_rows = len(dataframe.index)

    # Set up forecasting model
    classifier_model = TackClassifier().load_model("../models/xgb_classifier.model")
   
    max_lag_order = 5
    forecast_window_in_seconds = 180
    forecaster_model = TackForecaster(classifier_model, max_lag_order, forecast_window_in_seconds)

    for index, row in dataframe.iterrows():
        tacking = forecaster_model.forecast_from_datapoint(row)
        print("Predicted tacking label: %d" % tacking)
        if index + forecast_window_in_seconds < dataframe_num_rows:
            true_tacking_label = dataframe.iloc(index + forecast_window_in_seconds)["Tacking"]
            print("True tacking label: %d\n" % true_tacking_label)
        else:
            print("NoGT tacking label\n")

