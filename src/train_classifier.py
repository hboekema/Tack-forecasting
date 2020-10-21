

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DataLoader import DataLoader
from ClassifierTrainer import ClassifierTrainer
from TackClassifier import TackClassifier
from StateForecaster import StateForecaster

from preprocessing import dataframe_to_dmatrix_for_classifier_training, preprocess_dataframe_for_forecasting_training
from xgboost import plot_tree


if __name__ == "__main__":
    dataframe = DataLoader(datapath="../data/train/train_data.csv").load_dataframe_from_datapath()
    print(dataframe.dtypes)
    dataframe = dataframe.convert_dtypes(convert_integer=False)
    print(dataframe.dtypes)

    train_dmatrix, val_dmatrix = dataframe_to_dmatrix_for_classifier_training(dataframe)

    classifier_model_params = {
            'colsample_bynode': 0.8,
            'learning_rate': 1,
            'max_depth': 6,
            'num_parallel_tree': 20,
            'objective': 'binary:logistic',
            'eval_metric': 'error',
            'subsample': 0.8,
            #'tree_method': 'gpu_hist'
            }
    classifier = TackClassifier(classifier_model_params)

    eval_watchlist = [(train_dmatrix, 'train'), (val_dmatrix, 'val')]
    example_training_params = {'evals': eval_watchlist, 'num_boost_round': 40}

    classifier_model_trainer = ClassifierTrainer(classifier)
    classifier_model = classifier_model_trainer.fit(train_dmatrix, **example_training_params)

    print(classifier_model.booster.get_score(importance_type='gain'))

    classifier_model.save_model("../models/xgb_classifier.model")

    #plot_tree(classifier_model)
    #plt.show()
    classifier_model.plt()
