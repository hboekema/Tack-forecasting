


import pandas as pd

from DataLoader import DataLoader
from StateForecaster import StateForecaster
from preprocessing import preprocess_dataframe_for_forecasting_training



if __name__ == "__main__":
    dataframe = DataLoader().load_dataframe_from_datapath()

    forecaster_dataframe = preprocess_dataframe_for_forecasting_training(dataframe)
    print(forecaster_dataframe)
    forecaster_model = StateForecaster(forecaster_dataframe)
    result = forecaster_model.fit(5)
    print(result.summary())

    lag_order = result.k_ar
    print(lag_order)
    forecast = result.forecast(forecaster_dataframe.values[-lag_order:], steps=300)
    df_forecast = pd.DataFrame(forecast, index=forecaster_dataframe.index[-300:], columns=forecaster_dataframe.columns)
    print(df_forecast)

