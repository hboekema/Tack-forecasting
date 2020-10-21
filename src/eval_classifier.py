

import numpy as np

from DataLoader import DataLoader
from TackClassifier import TackClassifier

from preprocessing import preprocess_dataframe_for_classifier_training


if __name__ == "__main__":
    dataframe = DataLoader("../data/test/test_data.csv").load_dataframe_from_datapath()
    dataframe = dataframe.convert_dtypes(convert_integer=False)
    test_dmatrix = preprocess_dataframe_for_classifier_training(dataframe)

    classifier_model = TackClassifier()
    classifier_model.load_model("../models/xgb_classifier.model")

    result = classifier_model.eval(test_dmatrix)
    print(result)

    preds = classifier_model.predict(test_dmatrix)
    preds = [int(pred > 0.5) for pred in preds]
    gt_labels = dataframe["Tacking"].values

    true_preds = ~np.logical_xor(gt_labels, preds)
    accuracy = float(true_preds.sum())/float(len(true_preds))
    print("Accuracy: %.03f" % accuracy)
