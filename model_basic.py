#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 21:37:21 2020

@author: yangzhang
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
## prepping data for training
train = pd.read_csv('./data/train_final.csv')
test = pd.read_csv('./data/test_final.csv')
testids = test['PassengerId'].copy()

train = train.drop(['PassengerId'], axis = 1)
test = test.drop(['PassengerId'], axis = 1)

y = train['Survived'].copy()
X = train.drop('Survived', axis = 1)

##1. base algorithm benchmark
## split
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.3, random_state = 0)
classifiers = ['Naive Bayes', 'Perceptron', 'Logistic Regression', 'KNN', 
               'Decision Tree', 'Random Forests', 'Linear SVM', 'rbf SVM']
models = [GaussianNB(), Perceptron(class_weight='balanced'), LogisticRegression(), KNeighborsClassifier(),
          DecisionTreeClassifier(), RandomForestClassifier(n_estimators=100),
          svm.SVC(kernel='linear'), svm.SVC(kernel='rbf')]

def get_score(classifier, model, train_X, val_X, train_y, val_y):
    model.fit(train_X, train_y)
    return print('The accuracy score for', classifier, 'is', model.score(val_X, val_y))
 
for classifier, model in zip(classifiers, models):
    get_score(classifier, model, train_X, val_X, train_y, val_y)
    
##2. confusion matrix for base algorithms
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

f,ax = plt.subplots(nrows=3, ncols=3, 
                    sharex=True, sharey=True,
                    figsize=(12,10))
f.suptitle("Confusion Matrix for Base Classifiers", fontsize=16, fontweight='bold')
f.text(0.5, 0.06, 'True Class', ha='center', color='red', fontsize = 16)
f.text(0.06, 0.5, 'Predicted Class', va='center',
       fontsize = 16, color='red', rotation='vertical')

y_pred = cross_val_predict(GaussianNB(), X, y)
sns.heatmap(confusion_matrix(y, y_pred),ax = ax[0,0], 
            cmap = 'Set3', annot=True, fmt='2.0f')
ax[0,0].set_title('Naive Bayes')

y_pred = cross_val_predict(Perceptron(), X, y)
sns.heatmap(confusion_matrix(y, y_pred),ax = ax[0,1], 
            cmap = 'Set3', annot=True, fmt='2.0f')
ax[0,1].set_title('Perceptron')

y_pred = cross_val_predict(LogisticRegression(), X, y)
sns.heatmap(confusion_matrix(y, y_pred),ax=ax[0,2], 
            cmap = 'Set3', annot=True, fmt='2.0f')
ax[0,2].set_title('Logistic Regression')

y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=2), X, y)
sns.heatmap(confusion_matrix(y, y_pred),ax=ax[1,0], 
            cmap = 'Set3', annot=True, fmt='2.0f')
ax[1,0].set_title('KNN')

y_pred = cross_val_predict(DecisionTreeClassifier(), X, y)
sns.heatmap(confusion_matrix(y, y_pred),ax=ax[1,1], 
            cmap = 'Set3', annot=True,fmt='2.0f')
ax[1,1].set_title('Decision Tree')

y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100), X, y)
sns.heatmap(confusion_matrix(y, y_pred),ax=ax[1,2], 
            cmap = 'Set3', annot=True, fmt='2.0f')
ax[1,2].set_title('Random Forests')

y_pred = cross_val_predict(svm.SVC(kernel='linear'), X, y)
sns.heatmap(confusion_matrix(y, y_pred), ax=ax[2,0], 
            cmap = 'Set3', annot=True,fmt='2.0f')
ax[2,0].set_title('Linear SVM')

y_pred = cross_val_predict(svm.SVC(kernel='rbf'), X, y)
sns.heatmap(confusion_matrix(y, y_pred), ax=ax[2,1],
            cmap = 'Set3', annot=True, fmt='2.0f')
ax[2,1].set_title('rbf SVM')

plt.subplots_adjust(hspace=0.2,wspace=0.2)
f.subplots_adjust(top=0.92)
ax[2,2].remove()

plt.savefig('./figures/basicModelconfusionMatrix.png', dpi=150)
plt.show()

##2. stratified 10-fold cross validation on total training data, no split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

#folds = StratifiedKFold(n_splits = 10)
folds = StratifiedKFold(n_splits=10, random_state=46, shuffle=True)

def get_stratifiedCV_score(classifier, model, X, y):
    cv_results = cross_val_score(model, X, y, cv = folds, scoring = "accuracy")
    return print('Average cv score for', classifier, 'is', cv_results.mean())

for classifier, model in zip(classifiers, models):
    get_stratifiedCV_score(classifier, model, X, y)

##3. Hyperparameter tuning on top classifiers
#code source: https://www.kaggle.com/kenjee/titanic-project-example
from sklearn.model_selection import GridSearchCV
def mod_performance(model, classifier):
    print(classifier)
    print('Best Score: ' + str(model.best_score_))
    print('Best Parameters: ' + str(model.best_params_))
    

### a. logistic regression
lr = LogisticRegression()
param_grid = {'max_iter' : [2000],
              'penalty' : ['l1', 'l2'],
              'C' : np.logspace(-4, 4, 20),
              'solver' : ['liblinear']}

clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 10, verbose = True, n_jobs = -1)
best_clf_lr = clf_lr.fit(X, y)
mod_performance(best_clf_lr,'Logistic Regression')

### b. decision tree
dt = DecisionTreeClassifier()
param_grid ={'max_depth': [3, None],
             'max_features': [2,3],
             'criterion': ['gini', 'entropy']}
clf_dt = GridSearchCV(dt, param_grid = param_grid, cv = 10, verbose = True, n_jobs = -1)
best_clf_dt = clf_dt.fit(X, y)
mod_performance(best_clf_dt,'Decision tree')

### c. random forests
rf = RandomForestClassifier(random_state = 0)
param_grid =  {'n_estimators': range(100, 1000, 100),
               'criterion':['gini','entropy'],
               'bootstrap': [True],
               'max_depth': [15, 20, 25],
               'max_features': ['auto','sqrt', 8],
               'min_samples_leaf': [2,3],
               'min_samples_split': [2,3]
               }
clf_rf = GridSearchCV(rf, param_grid = param_grid, cv = 10, verbose = True, n_jobs = -1)
best_clf_rf = clf_rf.fit(X, y)
mod_performance(best_clf_rf,'Random Forest')

### d. rbf SVM
svc = svm.SVC(probability = True)
param_grid = tuned_parameters = {'kernel': ['rbf'], 
                                  'gamma': [0.1, 0.5, 1, 2, 5, 10],
                                  'C': [0.1, 1, 10, 100, 1000]}
clf_svc = GridSearchCV(svc, param_grid = param_grid, cv = 10, verbose = True, n_jobs = -1)
best_clf_svc = clf_svc.fit(X, y)
mod_performance(best_clf_svc,'SVC')

##4. Another round of 10-fold cross validation on tuned parameters
tuned_classifiers = ['Logistic Regression', 'Decision Tree', 'Random Forests', 'rbf SVM']
tuned_models = [LogisticRegression(C=4.2813, max_iter=2000, penalty='l2', solver='liblinear'), 
                DecisionTreeClassifier(criterion='gini', max_depth=None, max_features=3), 
                RandomForestClassifier(n_estimators=200, bootstrap=True,
                                       criterion='gini', max_depth=15, max_features=8,
                                       min_samples_leaf=2, min_samples_split=2),
                svm.SVC(kernel='rbf', C=1, gamma=0.1)]

tuned = [best_clf_lr, best_clf_dt, best_clf_rf, best_clf_svc]

for classifier, model in zip(tuned_classifiers, tuned_models):
    get_stratifiedCV_score(classifier, model, X, y)
#Average cv score for Logistic Regression is 0.7980024968789013
#Average cv score for Decision Tree is 0.8081023720349563
#Average cv score for Random Forests is 0.817116104868914
#Average cv score for rbf SVM is 0.8227091136079899


##submission
### random forest: score 0.75837
predictions = best_clf_rf.predict(test)
output = pd.DataFrame({'PassengerId': testids, 'Survived': predictions})
output.to_csv('./submission/submissionRF.csv', index=False)
## logistic regression: score 0.76555
predictions = best_clf_lr.predict(test)
output = pd.DataFrame({'PassengerId': testids, 'Survived': predictions})
output.to_csv('./submission/submissionLR.csv', index=False)
## rbf SVM: score 0.77272
predictions = best_clf_svc.predict(test)
output = pd.DataFrame({'PassengerId': testids, 'Survived': predictions})
output.to_csv('./submission/submissioSVMrbf.csv', index=False)


#Ensemble classifiers
### Boosting
### 1. XGBoost: eXtrem Gradient Boosting 
import xgboost as xg
from xgboost import XGBClassifier
xgboost = xg.XGBClassifier()
result = cross_val_score(xgboost, X, y, cv=10, scoring='accuracy')
print('The cross validated score for XGBoost is:',result.mean())

xgb = XGBClassifier(random_state = 0)
param_grid = {
    'n_estimators': [100, 200, 300, 450,500,550],
    'colsample_bytree': [0.75,0.8,0.85],
    'max_depth': [None],
    'reg_alpha': [1],
    'reg_lambda': [2, 5, 10],
    'subsample': [0.55, 0.6, .65],
    'learning_rate':[0.5],
    'gamma':[.5,1,2],
    'min_child_weight':[0.01],
    'sampling_method': ['uniform']
}
clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = 10, verbose = True, n_jobs = -1)
best_clf_xgb = clf_xgb.fit(X, y)
mod_performance(best_clf_xgb,'XGB')
#### submission (submission score: 0.7488)
y_hat_xgb = best_clf_xgb.best_estimator_.predict(test).astype(int)
xgb_submission = {'PassengerId': testids, 'Survived': y_hat_xgb}
submission_xgb = pd.DataFrame(data=xgb_submission)
submission_xgb.to_csv('./submission/submissionXGB.csv', index=False)

### 2. AdaBoost (adaptive boosting)
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(random_state = 0)
result = cross_val_score(ada, X, y, cv=10, scoring='accuracy')
print('The cross validated score for AdaBoost is:',result.mean()) ##0.8069

param_grid = {
    'n_estimators': list(range(100,1100,100)),
    'learning_rate': [0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
    }

clf_ada = GridSearchCV(ada, param_grid = param_grid, 
                       cv=10, verbose=True, n_jobs = -1)
best_clf_ada = clf_ada.fit(X, y)
mod_performance(best_clf_ada, 'AdaBoost') ##0.8215
#### submission (0.78468, best score)
y_hat_ada = best_clf_ada.best_estimator_.predict(test).astype(int)
ada_submission = {'PassengerId': testids, 'Survived': y_hat_ada}
submission_ada = pd.DataFrame(data=ada_submission)
submission_ada.to_csv('./submission/submissionAdaBoost.csv', index=False)

### 3. CatBoost
import catboost as cb
cat = cb.CatBoostClassifier()
param_grid = {
    'iterations': [5, 10, 15, 20, 25, 50, 100],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [3, 5, 7, 9, 11, 13]}

clf_cat = GridSearchCV(cat, param_grid = param_grid,
                       cv=10, verbose=True, n_jobs = -1)
best_clf_cat = clf_cat.fit(X, y)
mod_performance(best_clf_cat, 'CatBoost') ##0.83277

#### submission (0.77511)
y_hat_cat = best_clf_cat.best_estimator_.predict(test).astype(int)
cat_submission = {'PassengerId': testids, 'Survived': y_hat_cat}
submission_cat = pd.DataFrame(data=cat_submission)
submission_cat.to_csv('./submission/submissionCatBoost.csv', index=False)

### Voting: hard and soft
from sklearn.ensemble import VotingClassifier
best_lr = best_clf_lr.best_estimator_
best_svc = best_clf_svc.best_estimator_
best_rf = best_clf_rf.best_estimator_
best_xgb = best_clf_xgb.best_estimator_

voting_clf_hard = VotingClassifier(
    estimators = [('Logistic regression', best_lr),
                  ('rbf SVM', best_svc),
                  ('Random forests', best_rf),
                  ('XGBoost', best_xgb)],
    voting='hard')

voting_clf_soft = VotingClassifier(
    estimators = [('Logistic regression', best_lr),
                  ('rbf SVM', best_svc),
                  ('Random forests', best_rf),
                  ('XGBoost', best_xgb)],
    voting='soft')

print('voting_clf_hard :', cross_val_score(voting_clf_hard,X,y,cv=10))
print('voting_clf_hard mean :', cross_val_score(voting_clf_hard,X,y,cv=10).mean()) #0.8339

print('voting_clf_soft :',cross_val_score(voting_clf_soft,X,y,cv=10))
print('voting_clf_soft mean :',cross_val_score(voting_clf_soft,X,y,cv=10).mean()) #0.8204

##give weights in soft voting
params = {'weights' : [[1,1,1,1],[1,2,1,1],[1,1,2,1],[1,1,1,2],
                       [2,1,1,1],[2,2,1,1],[1,1,2,2],[2,1,2,1]]}

vote_weight = GridSearchCV(voting_clf_soft, param_grid = params, cv = 10, verbose = True, n_jobs = -1)
best_clf_weight = vote_weight.fit(X,y)
mod_performance(best_clf_weight,'VC Weights') ##0.8283

###submission 
###################
voting_clf_sub = best_clf_weight.best_estimator_.predict(test).astype(int)
vote_submission = {'PassengerId': testids, 'Survived': voting_clf_sub}
submission_vote = pd.DataFrame(data=vote_submission)
submission_vote.to_csv('./submission/submissionSoftvoteWeight.csv', index=False) ##0.77511

###################
voting_clf_hard.fit(X, y)
y_hat_vc_hard = voting_clf_hard.predict(test).astype(int)
hardvote_submission = {'PassengerId': testids, 'Survived': y_hat_vc_hard}
submission_hardvote = pd.DataFrame(data=hardvote_submission)
submission_hardvote.to_csv('./submission/submissionhardvote.csv', index=False)  ##0.77033

##Feature importance plots
f,ax = plt.subplots(nrows=2, ncols=2, figsize=(14,10))
f.suptitle("Feature Importance in Tuned Models with Best Performance", 
           fontsize=16, fontweight='bold')

best_lr = best_clf_lr.best_estimator_.fit(X, y)
feat_importances = pd.Series(best_lr.coef_[0], index=X.columns)
pd.Series(feat_importances, X.columns).sort_values(ascending=True).plot.barh(width=0.8, ax=ax[0,0], color='plum')
ax[0,0].set_title('Logistic Regression')

best_df = best_clf_dt.best_estimator_.fit(X, y)
feat_importances = pd.Series(best_df.feature_importances_, index=X.columns)
pd.Series(feat_importances, X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,1], color='plum')
ax[0,1].set_title('Decision Tree')

best_rf = best_clf_rf.best_estimator_.fit(X, y)
feat_importances = pd.Series(best_rf.feature_importances_, index=X.columns)
pd.Series(feat_importances, X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0], color='plum')
ax[1,0].set_title('Random Forests')

best_ada = best_clf_ada.best_estimator_.fit(X, y)
feat_importances = pd.Series(best_ada.feature_importances_, index=X.columns)
pd.Series(feat_importances, X.columns).sort_values(ascending=True).plot.barh(width=0.8, ax=ax[1,1], color='plum')
ax[1,1].set_title('AdaBoost')

f.tight_layout()
plt.savefig('./figures/featureImportance.png', dpi=150)
plt.show()





