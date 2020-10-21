

import xgboost as xgb
import matplotlib.pyplot as plt


class TackClassifier:
    def __init__(self, params={}):
        self.booster = None
        self.params = params

    def train(self, dtrain, *args, **kwargs):
        self.booster = xgb.train(self.params, dtrain, *args, **kwargs)
        return self.booster

    def cv(self, dtrain, *args, **kwargs):
        eval_history = xgb.cv(self.params, dtrain, *args, **kwargs)
        return eval_history

    def predict(self, data):
        assert self.booster is not None
        return self.booster.predict(data)

    def eval(self, data):
        assert self.booster is not None
        return self.booster.eval(data)

    def save_model(self, path):
        self.booster.save_model(path)
    
    def load_model(self, path):
        self.booster = xgb.Booster()
        self.booster.load_model(path)

    def plt(self):
        if self.booster is not None:
            xgb.plot_tree(self.booster)
            plt.show()
