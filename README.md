# credit-risk-classification

## Overview of the Analysis

The purpose of this project is to analyze a dataset of historical lending activity from a peer-to-peer lending services company in order to build a model that can identify the creditworthiness of borrowers. 

The dataset to analyze includes ***features*** such as the loan size, interest rate, borrower income, debt to income and total debt. And the dataset has one field, the loan status that will be used as the  ***outcome***.  The task is to predict whether the outcome is "healty" (0) or "high-risk" (1) for each loan application.

### Stages of the machine learning process

1. First, we will read the data file into a python DataFrame and use the "train_test_split" function of the scikit-learn library to split our datasetinto the training and testing sets to use for this exercise 
2. Then, we will create a Logistic Regression model and train it using the "train" data segment. 
3. Next, after our model has been trained, we will then make it predict how good are the loans in the set aside testing segment. 
4. And lastly, we will look into the performance of the model by looking at the balance_accuracy_score, a confusion matrix, and a classification report to evaluate the results of the model.

**Core functions and libraries for this work:*** 

Library: ***scikit-learn ***
Functions:  
- train_test_split
- LogisticRegression
- balance_accuracy_score
- confusion_matrix
- classification_report


In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

## Results

- The Balance Accuracy Score was 94%, which indicates ...
- The Confusion Matrix shows the following ...
- The Classification Report is as follows ...
  - describe precision, recall, etc.

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

In summary, the data point that best describes the outcome is ...

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.
