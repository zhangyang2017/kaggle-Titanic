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
  - Final features:
  ![](https://github.com/zhangyang2017/kaggle-Titanic/blob/master/figures/allFeatures.png)
  - some numerical feature (training set)
    - 891 samples, representing 40% (891 in 2224) of passengers boarded the Titanic;
    - survival rate ~ 38% (actual survival rate on Titanic is 32%)
    - the majority of passengers traveled alone (~75%)
  - data visualization, feature relationship exploration
    - `pivot tables` for quick assumptions
    - `histogram` `correlation heatmap` `kernal density estimation plot` to confirm assumptions
    - which features might contribute significantly to the model
      ![after some feature engineering](https://github.com/zhangyang2017/kaggle-Titanic/blob/master/all_fetures.png)
  - data wrangling
    - missing data imputation
      - This is the part where I can be creative. I was inspired by [this notebook][2] that instead of filling missing values with mean, median or mode, I can try a few more things. For example, I have two missing values under `Embarked` feature. Both of them paid 80$ for their first class tickets.
      ```
      X[X.Embarked.isnull()]
      ```
      ![temp](https://github.com/zhangyang2017/kaggle-Titanic/blob/master/temp.jpg)
        Using visualization, I can actually see that at C, the median Fair people paid for first class ticket is closer to 80. So replacing the two missing values with 'C' seems a reasonable guess. And a lot more fun :stuck_out_tongue:.
      ![temp2](https://github.com/zhangyang2017/kaggle-Titanic/blob/master/temp2.jpg)
    - `Age` feature had a lot more missing values. I never thought I can actually build a model to predict those values on other features such as titles and how much they paid for their tickets. Is it guaranteed more accurate than just replacing with mean? Probably not, but very cool!
    - Feature engineering
    - Drop not-so-useful features
- Model training
  - the idea of **benchmark**
- Model validation
  - basic models
    ![](https://github.com/zhangyang2017/kaggle-Titanic/blob/master/figures/basicModelconfusionMatrix2.png)
- prediction

## Afterthoughts <a id="3"></a>
https://matloff.wordpress.com/2015/09/29/unbalanced-data-is-a-problem-no-balanced-data-is-worse/
https://elitedatascience.com/imbalanced-classes
https://towardsdatascience.com/what-to-do-when-your-classification-dataset-is-imbalanced-6af031b12a36
https://sebastianraschka.com/faq/docs/scale-training-test.html
https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/43453

## Resources <a id="4"></a>

[1]: <https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74> (Hyperparameter Tuning the Random Forest in Python by Will Koehrsen)
[2]: <https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic#Part-3.-Visualization-and-Feature-Relations> (A Statistical Analysis & ML workflow of Titanic by Masum Rumi)


1. [Hyperparameter Tuning the Random Forest in Python](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74) by Will Koehrsen
2. [A Statistical Analysis & ML workflow of Titanic](https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic#Part-3.-Visualization-and-Feature-Relations) by Masum Rumi
3. [Titanic Project Example Walk Through](https://www.kaggle.com/kenjee/titanic-project-example/comments) by Ken Jee
4. [EDA To Prediction (DieTanic)](https://www.kaggle.com/ash316/eda-to-prediction-dietanic) by Ashwini Swain




https://www.kaggle.com/goldens/classification-review-with-python

https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic
https://www.kaggle.com/pavlofesenko/extending-titanic-dataset-using-wikipedia
https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8
https://elitedatascience.com/machine-learning-algorithms
https://www.kaggle.com/jirakst/titanic-auc-92
https://medium.com/i-like-big-data-and-i-cannot-lie/how-i-scored-in-the-top-9-of-kaggles-titanic-machine-learning-challenge-243b5f45c8e9



