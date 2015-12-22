# -*- coding: utf-8 -*-

__author__="Necati Demir <ndemir@demir.web.tr>"

from techniques import onego
from techniques import outoffolds


class StackingClassifier():

    ONEGO = 1
    OUTOFFOLDS = 2

    def __init__(self, base_classifiers, combiner, n=3, technique=ONEGO):
        if technique == self.ONEGO:
            self.stacking = onego.StackingClassifier(base_classifiers, combiner, n)
        elif technique == self.OUTOFFOLDS:
            self.stacking = outoffolds.StackingClassifier(base_classifiers, combiner, n)

    def fit(self, X, y):
        self.stacking.fit(X, y)

    def predict_proba(self, X):
        return self.stacking.predict_proba(X)

