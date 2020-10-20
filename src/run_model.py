


from TackClassifier import TackClassifier
from TackForecaster import TackForecaster

import argparse

parser = argparse.ArgumentParser(description="Run tack forecasting model on the datapoint provided")
parser.add_argument("datapoint", type=str, help="path to CSV datapoint to be used for prediction")

args = parser.parse_args()


if __name__ == "__main__":
    # Load datapoint
    datapoint = DataLoader(args["datapoint"]).load_dataframe_from_datapath()
    
    # Set up forecasting model
    classifier_model = TackClassifier().load_model("../models/xgb_classifier.model")
   
    max_lag_order = 5
    forecast_window_in_seconds = 180
    forecaster_model = TackForecaster(classifier_model, max_lag_order, forecast_window_in_seconds)

    tacking = forecaster_model.forecast_from_datapoint(datapoint)
    print("Tacking: %d" % tacking)


