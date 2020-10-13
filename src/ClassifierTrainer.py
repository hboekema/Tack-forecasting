


from TackClassifier import TackClassifier


class ClassifierTrainer:
    def __init__(self, objective='binary:logistic', **kwargs):
        self.objective = objective
        self.classifier_model_kwargs = kwargs
        self.classifier = TackClassifier(objective, **kwargs)

    def fit_with_kfold_validation(X, y, n_splits=n_splits):
        classifier_models = []

        kfolder = KFold(nsplits=n_splits, shuffle=True)
        for train_index, test_index in kfolder.split(X, y):
            classifier_model_k = TackClassifier(self.objective, **self.classifier_model_kwargs).fit(
                    X[train_index], y[train_index],
                    eval_set=[(X[train_index], y[train_index]), X[test_index], y[test_index]],
                    eval_metric='logloss'
                    verbose=True
                    )

            classifier_model_k_metric_score = classifier_model_k.evals_result()

            classifier_models.append(classifier_model_k, classifier_model_k_metric_score)

        # TODO: set best model

        

    def fit(X, y, n_splits=0):
        if n_splits is not None and n_splits > 1:
            self.fit_with_kfold_validation(X, y, n_splits=n_splits)
        else:
            self.classifier.fit(
                    X, y,
                    eval_set=[X, y],
                    eval_metric='logloss',
                    verbose=True
                    )

            classifier_model_k_metric_score = self.classifier.evals_result()

