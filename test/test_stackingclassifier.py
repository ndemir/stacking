# -*- coding: utf-8 -*-
__author__="Necati Demir <ndemir@demir.web.tr>"

import numpy as np
from StackingClassifier import StackingClassifier
from sklearn import linear_model, metrics
from sklearn import tree
import random

#Create an artificial data for demonstration
X = np.random.random((100, 10))
y = np.array([random.sample([0, 1],  1)[0]  for i in range(100)]).ravel()

#use ONEGO technique to create stacking model
stacking_classifier = StackingClassifier(
    base_classifiers=[
        linear_model.SGDClassifier(loss='log', random_state=0),
        linear_model.LogisticRegression(random_state=0),
        tree.DecisionTreeClassifier(random_state=0)
    ],
    combiner=linear_model.LogisticRegression(),
    technique=StackingClassifier.ONEGO
)

stacking_classifier.fit(X, y)
predicted_y_proba = stacking_classifier.predict_proba(X)
print metrics.roc_auc_score(y, predicted_y_proba)
# Since the dataset is meaningless, roc_auc_score will not produce meaningful result
# I am using it just to see, if there are any problems

#use OUTOFFOLDS technique to create stacking model
stacking_classifier = StackingClassifier(
    base_classifiers=[
        linear_model.SGDClassifier(loss='log', random_state=0),
        linear_model.LogisticRegression(random_state=0),
        tree.DecisionTreeClassifier(random_state=0)
    ],
    combiner=linear_model.LogisticRegression(),
    technique=StackingClassifier.OUTOFFOLDS
)

stacking_classifier.fit(X, y)
predicted_y_proba = stacking_classifier.predict_proba(X)
print metrics.roc_auc_score(y, predicted_y_proba)
# Since the dataset is meaningless, roc_auc_score will not produce meaningful result
# I am using it just to see, if there are any problems
