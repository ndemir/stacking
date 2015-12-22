# -*- coding: utf-8 -*-
__author__="Necati Demir <ndemir@demir.web.tr>"

import numpy as np
from sklearn import cross_validation
import copy

class StackingClassifier():
    """
    This class is a template of stacking method for classification.
    It only provides fit and predict_proba functions, and works with binary [0, 1] labels.
    predict_proba function returns the probability of label 1.
    To learn how to use, see test/test_stackingclassifier.py

    This stacking technique creates prediction dataset by taking the average of
    the out-of-fold predictors' predictions
    """
    def __init__(self, base_classifiers, combiner, n=3):
        self.base_classifiers = base_classifiers
        self.combiner = combiner
        self.n = n
        self.models = [[None for j in range(n)] for i in range(len(base_classifiers))]

    def fit(self, X, y):
        stacking_train = np.full(
            (np.shape(X)[0], len(self.base_classifiers)),
            np.nan
        )

        for model_no in range(len(self.base_classifiers)):
            cv = cross_validation.KFold(len(X), n_folds=self.n)
            for j, (traincv, testcv) in enumerate(cv):
                self.base_classifiers[model_no].fit(X[traincv, ], y[traincv])
                predicted_y_proba = self.base_classifiers[model_no].predict_proba(X[testcv,])[:, 1]
                stacking_train[testcv, model_no] = predicted_y_proba
                self.models[model_no][j] = copy.deepcopy(self.base_classifiers[model_no])
        self.combiner.fit(stacking_train, y)

    def predict_proba(self, X):
        stacking_predict_data = np.full(
            (np.shape(X)[0], len(self.base_classifiers)),
            np.nan
        )
        for model_no in range(len(self.base_classifiers)):
            dataset_blend = np.full(
                (np.shape(X)[0], len(self.models[model_no])),
                np.nan
            )
            for j in range(len(self.models[model_no])):
                dataset_blend[:, j] = self.models[model_no][j].predict_proba(X)[:, 1]
            stacking_predict_data[:, model_no] = dataset_blend.mean(1)
        return self.combiner.predict_proba(stacking_predict_data)[:, 1]

