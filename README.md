#Template for Stacking (Stacked Generalization) Ensemble Method


##What is Stacking?

Stacking, also known as Stacked Generalization, is an ensemble method where the goal is to combine the output of
machine learning algorithms with another machine learning algorithm.

##Simple Explanation About The Project 

StackingClassifier class acts as a proxy class that connects to one of two classes
that implement different stacking techniques. 

* In the technique called ONEGO, the prediction dataset is created in one go. Here is the steps in this technique:

    * 1.Split the train dataset into 3 parts: train1, train2, train3
    * 2.Fit a base classifier on train1 and create predictions for rest of the train dataset
    * 3.Fit the same classifier on train2 and create predictions for rest of the train dataset
    * 4.Fit the same classifier on train3 and create predictions for rest of the train dataset
    * 5.Fit the same classifier on the entire train set and create predictions for the prediction dataset
    * 6.Repeat steps 2,3,4 and 5 for each base classifiers
    * 7.Create a dataset called stacking_data that is output of steps 2,3 and 4
    * 8.Create a dataset called stacking_prediction that is output of step 5
    * 9.Fit the combiner classifier on stacking_data
    * 10.Use fitted combiner classifier for prediction on stacking_prediction

* In the technique called OUTOFFOLDS, the prediction dataset is created by taking the average of 
the out-of-fold predictors' predictions

Author: Necati Demir <ndemir@demir.web.tr>