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
    - This is the part where I can be creative. I was inspired by [this notebook][1] that instead of filling missing values with mean, median or mode, I can try a few more things. For example, I have two missing values under `Embarked` feature. Both of them paid 80$ for their first class tickets.
      ```
      X[X.Embarked.isnull()]
      ```
      ![temp](https://github.com/zhangyang2017/kaggle-Titanic/blob/master/temp.jpg)
      Using visualization, I can actually see that at C, the median Fair people paid for first class ticket is closer to 80. So replacing the two missing values with 'C' seems a reasonable guess. And a lot more fun :stuck_out_tongue:.
      ![temp2](https://github.com/zhangyang2017/kaggle-Titanic/blob/master/temp2.jpg)
    - `Age` feature had a lot more missing values. I never thought I can actually build a model to predict those values on other features such as titles and how much they paid for their tickets. Is it guaranteed more accurate than just replacing with mean? Probably not, but very cool!
- feature engineering
- model training
- model validation
- prediction


## Resources <a id="4"></a>

[1]: <https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic#Part-3.-Visualization-and-Feature-Relations> (A Statistical Analysis & ML workflow of Titanic by Masum Rumi)



https://www.kaggle.com/kenjee/titanic-project-example/comments

https://www.kaggle.com/startupsci/titanic-data-science-solutions
https://www.kaggle.com/ash316/eda-to-prediction-dietanic
https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic

https://matloff.wordpress.com/2015/09/29/unbalanced-data-is-a-problem-no-balanced-data-is-worse/
https://elitedatascience.com/imbalanced-classes


