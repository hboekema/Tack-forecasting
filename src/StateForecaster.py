

from statsmodels.tsa.vector_ar.var_model import VAR


class StateForecaster(VAR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

