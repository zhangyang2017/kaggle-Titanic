# Repo for the Kaggle "Titanic: machine learning from disaster"


:open_book:
- [Problem definition](#0)
- [Overview](#1)
- [Workflow](#2)
- [Afterthoughts](#3)
- [Resources that helped me learn](#4)

## Problem definition <a id="0"></a>
Train a :label: classification model to determine whether or not a passenger survived in the test set based on the survival information in the training set. More details are described :point_right: [here](https://www.kaggle.com/c/titanic).

## Overview <a id="1"></a>
Ah the famous Titanic Kaggle competition. I did not expect to have this much fun working through this project, honestly. In this classic classification problem, I got to try out a lot of algorithms, including `naive bayes`, `logistic regression`, `decision tree`, `KNN`, `random forest`, `SVM`, `XGBoost`, `soft voting classifier`. For me, `random forest` produced the highest accuracy at predicting the test dataset (78%). I did spend hours trying to increase the accuracy of my algorithm. I tried [hyperparameter tuning][1], engineering new and dropping low-impact features. Nothing I tried really helped with increasing the accuracy :sweat_smile:. Although, I am not good at fine-tuning or feature engineering yet.

Project duration: 2020/10/21 - 2020/10/23

## Workflow <a id="2"></a>
- EDA
  - some numerical feature (training set)
    - 891 samples, representing 40% (891 in 2224) of passengers boarded the Titanic;
    - survival rate ~ 38% (actual survival rate on Titanic is 32%)
    - the majority of passengers traveled alone (~75%)
  - data wrangling
    - missing data imputation
      - This is the part where I can be creative. I was inspired by [this notebook][2] that instead of filling missing values with mean, median or mode, I can try a few more things. 
    - `Age` feature had a lot more missing values. I never thought I can actually build a model to predict those values on other features such as titles and how much they paid for their tickets. Is it guaranteed more accurate than just replacing with mean? Probably not, but very cool!
    - Feature engineering
    - Drop not-so-useful features
  - data visualization, feature relationship exploration
    - `pivot tables` for quick assumptions
    - `histogram` `correlation heatmap` `kernal density estimation plot` to confirm assumptions
    - which features might contribute significantly to the model
  - Final features:
  ![](https://github.com/zhangyang2017/kaggle-Titanic/blob/master/figures/allFeatures.png)
     **Figure 1: Features used in model training.**
     
- Model summary
  
  | Classifier Accuracy |  Baseline  |  10-fold Stratified Cross Validation | Hyperparameter Tuned |
  |---------------------|:----------:|:------------------------------------:|:--------------------:|
  | naive bayes         |   0.7388   | 0.7665                               |NA                    |
  | Perceptron          |   0.7910   | 0.7227                               |NA                    |
  | linear SVM          |   0.7873   | 0.7834                               |NA                    |
  | KNN                 |   0.7910   | 0.7890                               |NA                    |
  | rbf SVM             |   0.8097   | 0.8216                               |0.8249                |
  | logistic regression |   0.8134   | 0.7991                               |0.8036                |
  | decision tree       |   0.8396   | 0.8093                               |0.8093                |
  | random forests      |   0.8545   | 0.8126                               |0.8250                |
**Table 1. Accuracy stats for base classifier benchmark, after k-fold cross validation, and after hyperparameter tuning for the top classifiers.**

  ![](https://github.com/zhangyang2017/kaggle-Titanic/blob/master/figures/basicModelconfusionMatrix.png)
**Figure 2. Confusion Matrix for base classifiers. Seems like Naive Bayes was very good at predicting survivors, whereas KNN was very good at predicting non-survivors.**

  ![](https://github.com/zhangyang2017/kaggle-Titanic/blob/master/figures/featureImportance.png)
**Figure 3. Feature importance plots for final classifiers.**

- Submission scores
  |          Model        |    Score    |  
  |-----------------------|:-----------:| 
  | AdaBoost              |   0.78468   | 
  |Voting (soft, weighted)|   0.77511   |
  | logistic regression   |   0.76555   | 
  | random forests        |   0.75837   | 

## Afterthoughts <a id="3"></a>
- the idea of **benchmark**
- thoughts on data imbalance
  - https://matloff.wordpress.com/2015/09/29/unbalanced-data-is-a-problem-no-balanced-data-is-worse/
  - https://elitedatascience.com/imbalanced-classes
  - https://towardsdatascience.com/what-to-do-when-your-classification-dataset-is-imbalanced-6af031b12a36
  - https://sebastianraschka.com/faq/docs/scale-training-test.html
  - https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/43453

## Resources <a id="4"></a>

[1]: <https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74> (Hyperparameter Tuning the Random Forest in Python by Will Koehrsen)
[2]: <https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic#Part-3.-Visualization-and-Feature-Relations> (A Statistical Analysis & ML workflow of Titanic by Masum Rumi)


1. [Hyperparameter Tuning the Random Forest in Python](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74) by Will Koehrsen
2. [A Statistical Analysis & ML workflow of Titanic](https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic#Part-3.-Visualization-and-Feature-Relations) by Masum Rumi
3. [Titanic Project Example Walk Through](https://www.kaggle.com/kenjee/titanic-project-example/comments) by Ken Jee
4. [EDA To Prediction (DieTanic)](https://www.kaggle.com/ash316/eda-to-prediction-dietanic) by Ashwini Swain
5. [Predicting the Survival of Titanic Passengers](https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8) by Niklas Donges

## Other interesting posts <a id="5"></a>

1. [Kaggle dataset extended with Wikipedia](https://www.kaggle.com/pavlofesenko/extending-titanic-dataset-using-wikipedia) by Pavlo Fesenko.
2. https://www.kaggle.com/goldens/classification-review-with-python
3. https://elitedatascience.com/machine-learning-algorithms
