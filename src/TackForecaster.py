

from TackClassifier import TackClassifier
from StateForecaster import StateForecaster
from preprocessing import preprocess_dataframe_for_classification, preprocess_dataframe_for_forecasting


class TackForecaster:
    def __init__(self, classifying_model, max_lag_order, forecast_window_in_seconds):
        self.forecasting_model = None
        self.classifying_model = classifying_model

        self.forecast_window_in_seconds = forecast_window_in_seconds
        self.max_lag_order = max_lag_order
        self.current_history_window = []
        self.preprocessed_history_dataframe = None

        self.tacking = None
        self.tack_in_future = None
        self.lag_order = None

        self.columns_required_by_model = None
        self.datapoint_columns = None

    def empty_current_history_window(self):
        self.current_history_window = []

    def add_datapoint_to_history_window(self, datapoint):
        if self.datapoint_columns = None
            self.datapoint_columns = datapoint.columns
        self.current_history_window.append(datapoint.values)
   
    def check_for_tacking(self, state):
        new_tacking_label = self.classify_state(state)
        
        self.check_for_tack_change(new_tacking_label)
        self.tacking = new_tack_label

    def check_for_tack_change(self, new_tacking_label):
        tack_change = bool(self.tacking) ^ bool(new_tacking_label)

        if tack_change:
            self.empty_current_history_window()

    def prepare_history_for_forecasting(self):
        assert self.columns_required_by_model is not None

        history_dataframe = pd.DataFrame(self.current_history_window, columns=self.datapoint_columns)
        self.preprocessed_history_dataframe = preprocess_dataframe_for_forecasting(history_dataframe)

    def fit_forecaster(self):
        forecaster = StateForecaster(self.current_history_window)
        self.forecasting_model = forecaster.fit(self.max_lag_order)
        
        self.lag_order = self.forecasting_model.k_ar

    def forecast_state(self):
        assert self.preprocessed_history_dataframe is not None
        assert len(self.preprocessed_history_dataframe.values) >= self.lag_order

        history_for_forecasting = self.preprocessed_history_dataframe.values[-self.lag_order:]
        forecasted_states_array = self.forecasting_model.forecast(history_for_forecasting, steps=self.forecast_window_in_seconds)
        forecasted_states = pd.DataFrame(forecasted_states_array, columns=self.colu,mns_required_by_model)

        forecasted_state = forecast_states[-1]    # only return final (i.e. at end of forecast window) state
        return forecasted_state

    def classify_state(self, state):
        preprocessed_state = preprocess_dataframe_for_classification(state)
        if self.columns_required_by_model is None:
            self.columns_required_by_model = preprocessed_state.columns

        return self.classifying_model.predict(preprocessed_state)

    def forecast_from_datapoint(self, datapoint):
        self.check_for_tacking(datapoint)

        self.add_datapoint_to_history_window(datapoint)
        self.fit_forecaster()
        
        if len(self.current_history_window) >= self.lag_order:    
            self.prepare_history_for_forecasting()
            forecasted_state = self.forecast_state()
            self.tack_in_future = self.classify_state(forecasted_state)

        return self.tack_in_future

