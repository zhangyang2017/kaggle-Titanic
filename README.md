# Repo for the Kaggle ["Titanic: machine learning from disaster"](https://www.kaggle.com/c/titanic/overview)

> #classification

Table of contents
- [Overview](#1)
- [Workflow](#2)
- [One more thought](#3)
- [Resources that helped me learn](#4)


## Overview <a id="1"></a>
Ah the famous Titanic Kaggle competition. I did not expect to have this much fun working through this project, honestly. In this classic classification problem, I got to try out a lot of algorithms, including `naive bayes`, `logistic regression`, `decision tree`, `KNN`, `random forest`, `SVM`, `XGBoost`, `soft voting classifier`. For me, `random forest` produced the highest accuracy at predicting the test dataset (78%). I did spend hours trying to increase the accuracy of my algorithm. I tried hyperparameter tuning, engineering new and dropping low-impact features. Nothing I tried really helped with increasing the accuracy :sweat_smile:. Although, I am not good at fine-tuning or feature engineering yet.

Project duration: 2020/10/21 - 2020/10/23

## Workflow <a id="2"></a>
- EDA
  - data visualization, feature relationship exploration
  - missing data imputation
    - This is the part where I can be creative. I was inspired by [this notebook][1] that I can actually build a model to predict missing age values on features such as their titles and how much they paid for their tickets. I thought this is very cool.
- feature engineering
- model training
- model validation
- prediction


## Resources <a id="3"></a>

[1]: <https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic#Part-3.-Visualization-and-Feature-Relations> "Kaggle Notebook"



https://www.kaggle.com/kenjee/titanic-project-example/comments

https://www.kaggle.com/startupsci/titanic-data-science-solutions
https://www.kaggle.com/ash316/eda-to-prediction-dietanic
https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic

https://matloff.wordpress.com/2015/09/29/unbalanced-data-is-a-problem-no-balanced-data-is-worse/
https://elitedatascience.com/imbalanced-classes


