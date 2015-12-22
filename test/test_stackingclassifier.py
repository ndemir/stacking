# -*- coding: utf-8 -*-
__author__="Necati Demir <ndemir@demir.web.tr>"

import numpy as np
from StackingClassifier import StackingClassifier
from sklearn import linear_model
from sklearn import tree
import random

#Create an artificial data for demonstration
X = np.random.random((100, 10))
y = np.array([random.sample([0, 1],  1)[0]  for i in range(100)]).ravel()

stacking_classifier = StackingClassifier(
    base_classifiers=[
        linear_model.SGDClassifier(loss='log'),
        linear_model.LogisticRegression(),
        tree.DecisionTreeClassifier()
    ],
    combiner=linear_model.LogisticRegression()
)

stacking_classifier.fit(X, y)
predicted_y_proba = stacking_classifier.predict_proba(X)