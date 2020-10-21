

import pandas as pd

from TackClassifier import TackClassifier
from StateForecaster import StateForecaster
from preprocessing import dataframe_to_dmatrix_for_classification, preprocess_dataframe_for_forecasting


class TackForecaster:
    def __init__(self, classifying_model, max_lag_order, forecast_window_in_seconds):
        self.forecasting_model = None
        self.classifying_model = classifying_model

        self.forecast_window_in_seconds = forecast_window_in_seconds
        self.max_lag_order = max_lag_order
        assert self.forecast_window_in_seconds > 0
        assert self.max_lag_order > 0

        self.current_history_window = []
        self.preprocessed_history_dataframe = None
        self.preprocessed_history_dataframe_no_mean = None

        self.forecasted_states = None
        self.forecasted_state = None
        self.lag_order = None

        self.tacking = None
        self.tack_in_future = None

        self.datapoint_columns = None

    def empty_current_history_window(self):
        self.current_history_window = []

    def add_datapoint_to_history_window(self, datapoint):
        if self.datapoint_columns is None:
            self.datapoint_columns = datapoint.columns
        self.current_history_window.extend(datapoint.values)

    def check_for_tacking(self, state):
        preprocessed_state = dataframe_to_dmatrix_for_classification(state)
        new_tacking_label = self.classify_state(preprocessed_state)
        
        self.check_for_tack_change(new_tacking_label)
        self.tacking = new_tacking_label

    def check_for_tack_change(self, new_tacking_label):
        tack_change = bool(self.tacking) ^ bool(new_tacking_label)

        if tack_change:
            self.empty_current_history_window()

    def _subtract_rolling_means_from_history(self):
        assert self.preprocessed_history_dataframe is not None

        history_rolling_means = self.preprocessed_history_dataframe.rolling(self.max_lag_order).mean()
        self.preprocessed_history_dataframe_no_mean = self.preprocessed_history_dataframe - history_rolling_means
        self.preprocessed_history_dataframe_no_mean.dropna(inplace=True)

    def _add_rolling_mean_to_forecast(self):
        assert self.forecasted_state is not None
        assert self.forecasted_states is not None
        assert self.preprocessed_history_dataframe is not None

        pass

    def _prepare_history_for_forecasting(self):
        history_dataframe = pd.DataFrame(self.current_history_window, columns=self.datapoint_columns)
        self.preprocessed_history_dataframe = preprocess_dataframe_for_forecasting(history_dataframe)
        #self._subtract_rolling_means_from_history()
        self.preprocessed_history_dataframe_no_mean = self.preprocessed_history_dataframe

    def _prepare_forecast_for_classifying(self):
        assert self.forecasted_state is not None

        self.forecasted_state = dataframe_to_dmatrix_for_classification(self.forecasted_state)

    def fit_forecaster(self):
        assert self.preprocessed_history_dataframe_no_mean is not None
        forecaster = StateForecaster(self.preprocessed_history_dataframe_no_mean)
        self.forecasting_model = forecaster.fit(self.max_lag_order, trend='nc')
        
        self.lag_order = self.forecasting_model.k_ar

    def forecast_state(self):
        assert self.preprocessed_history_dataframe_no_mean is not None
        assert len(self.preprocessed_history_dataframe_no_mean.index) >= self.lag_order

        history_for_forecasting = self.preprocessed_history_dataframe_no_mean.values[-self.lag_order:]
        forecasted_states_array = self.forecasting_model.forecast(history_for_forecasting, steps=self.forecast_window_in_seconds)
        self.forecasted_states = pd.DataFrame(forecasted_states_array, columns=self.preprocessed_history_dataframe_no_mean.columns)

        forecasted_state_as_series = self.forecasted_states.iloc[-1]
        forecasted_state = forecasted_state_as_series.to_frame().T.convert_dtypes(convert_integer=False)    # only return final (i.e. at end of forecast window) state
        return forecasted_state

    def classify_state(self, state):
        classifying_score = self.classifying_model.predict(state)
        return int(*classifying_score > 0.5)

    def forecast_from_datapoint(self, datapoint):
        self.check_for_tacking(datapoint)

        self.add_datapoint_to_history_window(datapoint)
        if len(self.current_history_window) > 2*self.max_lag_order:    
            self._prepare_history_for_forecasting()
            self.fit_forecaster()
        
            self.forecasted_state = self.forecast_state()
            self._add_rolling_mean_to_forecast()
            self._prepare_forecast_for_classifying()
            self.tack_in_future = self.classify_state(self.forecasted_state)

            return self.tacking
            #return self.tack_in_future
            
        else:
            return -1
