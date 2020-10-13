




def preprocess_dataframe_for_classification(dataframe):
    feature_column_names = ["CurrentSpeed", "CurrentDir", "TWS", "TWA", "AWS", "AWA", "Roll", "Pitch", "HoG", "HeadingTrue", "SoG", "VMG", "RudderAng", "Leeway", "ModePilote"]
    feature_dataframe = dataframe[feature_column_names]

    return feature_dataframe


def preprocess_dataframe_for_classifier_training(dataframe):
    X = preprocess_dataframe_for_classification(dataframe)
    y = dataframe["Tacking"]

    return X, y

