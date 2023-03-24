# credit-risk-classification

## Overview of the Analysis

The purpose of this project is to analyze a dataset of historical lending activity from a peer-to-peer lending services company in order to build a model that can identify the creditworthiness of borrowers. 

The dataset to analyze includes ***features*** such as the loan size, interest rate, borrower income, debt to income and total debt. And the dataset has one field, the loan status that will be used as the  ***outcome***.  The task is to predict whether the outcome is "healthy" (0) or "high-risk" (1) for each loan application.

### Stages of the machine learning process

1. First, we will read the data file into a python DataFrame and use the "train_test_split" function of the scikit-learn library to split our dataset into the training and testing data sets for this task 
2. Then, we will instantiate a Logistic Regression model and train it using the "training" dataset. By default, the training dataset is 75% of the data. 
3. Next, after our model has been trained, we will then make it predict how good are the loans in the set aside testing segment. We used the testing data to predict as this is the data the model hasn't seen in order to see how well it can predict it.
4. And lastly, we will look into the performance of the model by looking at the balance_accuracy_score, a confusion matrix, and a classification report to evaluate the results of the model.

**Core functions and libraries for this work:** 

Main Library: ***scikit-learn*** 

Library functions:  
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
After running the model against the testing data:
- The Balance Accuracy Score using the new or test data is 94%.
- The Confusion Matrix is as follows:


![image](https://user-images.githubusercontent.com/115383317/227534056-ef91b388-dd49-4b22-b585-1fee1a6ae594.png)



  The false numbers need to be as low as possible and 80 is the false positive, and 67 is the false negative.  Very low numbers compared to the rest.
  
- The Classification Report which shows the test results in evaluating the number of predicted occurrences shows:

![image](https://user-images.githubusercontent.com/115383317/227543602-7c5647ba-9e8c-4c22-927a-9fe5a49ef27c.png)


  - precision: ratio of correctly predicted positive observations to the total predicted positive observations i.e. out of all high-risk loans, how many actually were high-risk? ***Passing Score***
  - recall: ratio of correctly predicted observations out of all predicted observations (positive and negative) i.e. out of all loans, how many were correctly predicted as high-risk? ***Passing Score***
  - f1-score: one number to summarize precision and recall. An overall score. ***Passing Score***
  - accuracy: overall score 99.  ***Passing Score***
  

## Summary

In summary, the data point that best describes the outcome is ...

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.
