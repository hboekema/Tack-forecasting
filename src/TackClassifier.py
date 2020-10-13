

import xgboost


class TackClassifier(xgboost.XGBClassifier):
    def __init__(self, objective='binary:logistic', **kwargs):
        super().__init__(objective, **kwargs)
