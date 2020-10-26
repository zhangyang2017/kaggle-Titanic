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

## split
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.3, random_state = 0)

##1. Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
nb = GaussianNB()
nb.fit(train_X, train_y)
nb_pred = nb.predict(val_X)
print('The accuracy of the NaiveBayes is', metrics.accuracy_score(nb_pred, val_y))

##2. Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_X, train_y)
lr_pred=lr.predict(val_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(lr_pred, val_y))

##3. KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier() 
knn.fit(train_X, train_y)
knn_pred = knn.predict(val_X)
print('The accuracy of the KNN is', metrics.accuracy_score(knn_pred, val_y))

##check accuracies over n
a_index = list(range(1,11))
a = pd.Series()
x = [0,1,2,3,4,5,6,7,8,9,10]
for i in a_index:
    knn = KNeighborsClassifier(n_neighbors=i) 
    knn.fit(train_X, train_y)
    knn_pred = knn.predict(val_X)
    a = a.append(pd.Series(metrics.accuracy_score(knn_pred, val_y)))
plt.plot(a_index, a)
plt.xticks(x)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()
print('Accuracies for different values of n are:',a.values,'with the max value as ',a.values.max())

##4. Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(train_X, train_y)
dt_pred = dt.predict(val_X)
print('The accuracy of the Decision Tree is', metrics.accuracy_score(dt_pred, val_y))

##5. Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(train_X, train_y)
rf_pred = rf.predict(val_X)
print('The accuracy of the Random Forests is', metrics.accuracy_score(rf_pred, val_y))

##6. Linear SVM
from sklearn import svm
lsvm = svm.SVC(kernel='linear', C=0.1, gamma=0.1)
lsvm.fit(train_X, train_y)
lsvm_pred = lsvm.predict(val_X)
print('Accuracy for linear SVM is', metrics.accuracy_score(lsvm_pred, val_y))


##7. Radial SVM
rsvm = svm.SVC(kernel='rbf', C=1, gamma=0.1)
rsvm.fit(train_X, train_y)
rsvm_pred = rsvm.predict(val_X)
print('Accuracy for rbf SVM is', metrics.accuracy_score(rsvm_pred, val_y))

## Cross Validation
## K-fold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=10, random_state=22, shuffle=True) # k=10, split the data into 10 equal parts
xyz = []
accuracy = []
std = []
classifiers = ['Naive Bayes', 'Logistic Regression', 'KNN', 'Decision Tree',
               'Random Forest', 'Linear SVM', 'rbf SVM']
models = [GaussianNB(), LogisticRegression(), KNeighborsClassifier(n_neighbors=2),
          DecisionTreeClassifier(), RandomForestClassifier(n_estimators=100),
          svm.SVC(kernel='linear'), svm.SVC(kernel='rbf')]

for i in models:
    model = i
    cv_result = cross_val_score(model, train_X, train_y, cv = kfold, scoring = "accuracy")
    cv_result = cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_df = pd.DataFrame({'CV Mean':xyz,'Std':std}, index=classifiers)       
new_models_df

plt.subplots(figsize=(12,6))
box=pd.DataFrame(accuracy,index=[classifiers])
box.T.boxplot()

new_models_df['CV Mean'].plot.barh(width=0.8)
plt.title('Average CV Mean Accuracy')
fig=plt.gcf()
fig.set_size_inches(8,5)
plt.show()


## Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
#from sklearn.metrics import ConfusionMatrixDisplay

f,ax = plt.subplots(3, 3, figsize=(12,10))
#sns.set_palette(sns.color_palette("Set3"))

y_pred = cross_val_predict(GaussianNB(), X, y, cv=10)
sns.heatmap(confusion_matrix(y, y_pred),ax = ax[0,0], 
            cmap = 'Set3', annot=True, fmt='2.0f')
ax[0,0].set_title('Naive Bayes')

y_pred = cross_val_predict(LogisticRegression(), X, y, cv=10)
sns.heatmap(confusion_matrix(y, y_pred),ax=ax[0,1], 
            cmap = 'Set3', annot=True, fmt='2.0f')
ax[0,1].set_title('Logistic Regression')

y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=2), X, y, cv=10)
sns.heatmap(confusion_matrix(y, y_pred),ax=ax[0,2], 
            cmap = 'Set3', annot=True, fmt='2.0f')
ax[0,2].set_title('KNN')

y_pred = cross_val_predict(DecisionTreeClassifier(), X, y, cv=10)
sns.heatmap(confusion_matrix(y, y_pred),ax=ax[1,0], 
            cmap = 'Set3', annot=True,fmt='2.0f')
ax[1,0].set_title('Decision Tree')

y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100), X, y, cv=10)
sns.heatmap(confusion_matrix(y, y_pred),ax=ax[1,1], 
            cmap = 'Set3', annot=True, fmt='2.0f')
ax[1,1].set_title('Random Forests')

y_pred = cross_val_predict(svm.SVC(kernel='linear'), X, y, cv=10)
sns.heatmap(confusion_matrix(y, y_pred), ax=ax[1,2], 
            cmap = 'Set3', annot=True,fmt='2.0f')
ax[1,2].set_title('Linear SVM')

y_pred = cross_val_predict(svm.SVC(kernel='rbf'), X, y, cv=10)
sns.heatmap(confusion_matrix(y, y_pred), ax=ax[2,0],
            cmap = 'Set3', annot=True, fmt='2.0f')
ax[2,0].set_title('rbf SVM')

plt.subplots_adjust(hspace=0.2,wspace=0.2)
ax[2,1].remove()
ax[2,2].remove()
plt.show()




from sklearn import tree

from sklearn.preprocessing import StandardScaler


from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
#from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, accuracy_score
#from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV 
#from sklearn.model_selection import RandomizedSearchCV 
##########################################################






#creating trial RFC model
model_try = RandomForestClassifier(n_estimators=400, max_depth=5, random_state=42)
model_try.fit(train_X, train_y)
preds = model_try.predict(val_X)
print(accuracy_score(val_y, preds))

#feature importance of the algorithm
pd.Series(model_try.feature_importances_, index = train_X.columns).nlargest(12).plot(kind = 'barh',
                            figsize = (10, 10),title = 'Feature importance from Random Forest').invert_yaxis()


model = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
model.fit(X, y)
predictions = model.predict(test)
output = pd.DataFrame({'PassengerId': testids, 'Survived': predictions})
output.to_csv('./submission/submissionRFnew.csv', index=False)

## best thus far.








rf= RandomForestClassifier(random_state=1)

# grid searh to choose the best (combination of) hyperparameters
pg_rf={'n_estimators': [100,200,400],'max_depth': [20,40,50,60]}

gs_rf=GridSearchCV(estimator= rf,
               param_grid= pg_rf,
               scoring='accuracy',
               cv=2)

# nested cross validation combining grid search (inner loop) and k-fold cv (outter loop)
gs_rf_scores = cross_val_score(gs_rf, X, y, cv=5,scoring='accuracy', n_jobs=-1)

# fit, and fit with best estimator
gs_rf.fit(X, y)
gs_rf_best=gs_rf.best_estimator_
gs_rf_best.fit(X, y)

print('Train Accuracy:   {0:.1f}%'.format(gs_rf.score(X, y)*100))
print('CV Mean Accuracy: {0:.1f}%'.format(np.mean(gs_rf_scores)*100))
print('Test Accuracy:    {0:.1f}%'.format(gs_rf.score(xtest, ytest)*100))


gs_rf_best.fit(X, y)
predictions = gs_rf_best.predict(test)
output = pd.DataFrame({'PassengerId': testids, 'Survived': predictions})
output.to_csv('./submission/submissionRF3.csv', index=False)
















def correlation_heatmap(df, method):
    _ , ax = plt.subplots(figsize =(18, 16))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(method=method),
        cmap = colormap,
        square=True, 
        annot=True, 
        annot_kws={'fontsize':9 }
    )
    
    plt.title('Correlation Matrix', y=1.05, size=15)
    
    
plot = correlation_heatmap(X, 'pearson')
#plt.savefig('all_fetures.pdf')
#plt.show()

##drop low correlation features
train = train.drop(['Age', 'FamilySize_1', 'FamilySize_2', 'FamilySize_3', 'FamilySize_5', 'FamilySize_6', 'FamilySize_7', 'FamilySize_8',
                    'FamilySize_11', 'Pclass_2', 'Embarked_Q', 'age_group_adult', 'age_group_child',
                    'age_group_elderly', 'age_group_middleAged', 'age_group_senior', 'age_group_teenager',
                    'age_group_toddler', 'Pclass_3'], axis = 1)
test = test.drop(['Age', 'FamilySize_1', 'FamilySize_2', 'FamilySize_3', 'FamilySize_5', 'FamilySize_6', 'FamilySize_7', 'FamilySize_8',
                    'FamilySize_11', 'Pclass_2', 'Embarked_Q', 'age_group_adult', 'age_group_child',
                    'age_group_elderly', 'age_group_middleAged', 'age_group_senior', 'age_group_teenager',
                    'age_group_toddler', 'Pclass_3'], axis = 1)

correlation_heatmap(train, 'pearson')



svc=make_pipeline(StandardScaler(),SVC(random_state=1))
r=[0.0001,0.001,0.1,1,10,50,100]
PSVM=[{'svc__C':r, 'svc__kernel':['linear']},
      {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]
GSSVM=GridSearchCV(estimator=svc, param_grid=PSVM, scoring='accuracy', cv=2)
scores_svm=cross_val_score(GSSVM, X.astype(float), y, scoring='accuracy', cv=5)
np.mean(scores_svm)

model=GSSVM.fit(X, y)
pred=model.predict(test)
output=pd.DataFrame({'PassengerId':testids,'Survived':pred})
output.to_csv('./submission/svm_submission2.csv', index=False)







## scaling
std_scale = StandardScaler()
X_scale = std_scale.fit_transform(X)
test_scale = std_scale.transform(test)


#############################
######## naive Bayes ########
#############################
gnb = GaussianNB()
cv = cross_val_score(gnb, train_X, train_y, cv = 5)
print(cv)
print(cv.mean()) #0.694

#############################
#### logistic regression ####
#############################
logReg = LogisticRegression(max_iter = 2000)
cv = cross_val_score(logReg, train_X, train_y, cv = 5)
print(cv)
print(cv.mean()) #0.812

#############################
####### decision tree #######
#############################
dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt, train_X, train_y, cv=5)
print(cv)
print(cv.mean()) #0.809

#############################
############ KNN ############
#############################
knn = KNeighborsClassifier()
cv = cross_val_score(knn, train_X, train_y, cv=5)
print(cv)
print(cv.mean()) #0.776

#############################
####### random forest #######
#############################
rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf, train_X, train_y, cv=5)
print(cv)
print(cv.mean()) #0.805

#############################
############ SVM ############
#############################
svc = SVC(probability = True)
cv = cross_val_score(svc, X_scale, y, cv=5)
print(cv)
print(cv.mean()) #0.821
model = svc.fit(X_scale, y)
pred=model.predict(test_scale)
output=pd.DataFrame({'PassengerId':testids,'Survived':pred})
output.to_csv('./submission/svm_submissionNew.csv', index=False)

##another way

svc=make_pipeline(StandardScaler(),SVC(random_state=1))
r=[0.0001,0.001,0.1,1,10,50,100]
PSVM=[{'svc__C':r, 'svc__kernel':['linear']},
      {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]
GSSVM=GridSearchCV(estimator=svc, param_grid=PSVM, scoring='accuracy', cv=2)
scores_svm=cross_val_score(GSSVM, X.astype(float), y, scoring='accuracy', cv=5)
np.mean(scores_svm)

model=GSSVM.fit(X, y)
pred=model.predict(test)
output=pd.DataFrame({'PassengerId':testids,'Survived':pred})
output.to_csv('./submission/svm_submission.csv', index=False)

#############################
########## XGBoost ##########
#############################
xgb = XGBClassifier(random_state =1)
cv = cross_val_score(xgb, train_X, train_y, cv=5)
print(cv)
print(cv.mean()) #0.835

#############################
######## soft voting ########
#############################
voting_clf = VotingClassifier(estimators = [('lr',logReg),('knn',knn),('rf',rf),('gnb',gnb),('svc',svc),('xgb',xgb)], voting = 'soft')
cv = cross_val_score(voting_clf, train_X, train_y, cv=5)
print(cv)
print(cv.mean()) #0.823



voting_clf.fit(X_scale, y)
y_hat_base_vc = voting_clf.predict(test_scale).astype(int)
basic_submission = {'PassengerId': testids, 'Survived': y_hat_base_vc}
base_submission = pd.DataFrame(data=basic_submission)
base_submission.to_csv('./submission/base_submission.csv', index=False)

#######################################################################################

#simple performance reporting function
def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: ' + str(classifier.best_score_))
    print('Best Parameters: ' + str(classifier.best_params_))

xgb = XGBClassifier(random_state = 1)

param_grid = {
    'n_estimators': [450,500,550],
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

clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_xgb = clf_xgb.fit(X, y)
clf_performance(best_clf_xgb,'XGB')

y_hat_xgb = best_clf_xgb.best_estimator_.predict(test).astype(int)
xgb_submission = {'PassengerId': testids, 'Survived': y_hat_xgb}
submission_xgb = pd.DataFrame(data=xgb_submission)
submission_xgb.to_csv('./submission/xgb_submissionNew.csv', index=False)


model = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=42)
model.fit(train_X, train_y)
predictions = model.predict(val_X)

print(accuracy_score(val_y, predictions))


#######################################################################################
#accuracy_score(val_y, y_pred)

logReg.fit(X_scale,y)
test_preds = logReg.predict(test_scale)
#Submission
output = pd.DataFrame({'PassengerId': testids,
                       'Survived': test_preds})
output.to_csv('./submission/submission_logisticR.csv', index=False)


y_pred = logReg.predict(val_X)

# printing confision matrix
pd.DataFrame(confusion_matrix(val_y,y_pred),\
            columns=["Predicted Not-Survived", "Predicted Survived"],\
            index=["Not-Survived","Survived"] )
    
    

### random forest
# create param grid object 
forrest_params = dict(     
    max_depth = [n for n in range(9, 14)],     
    min_samples_split = [n for n in range(4, 11)], 
    min_samples_leaf = [n for n in range(2, 5)],     
    n_estimators = [n for n in range(10, 60, 10)],
)

# instantiate Random Forest model
forrest = RandomForestClassifier()
forest_cv = GridSearchCV(estimator=forrest, param_grid=forrest_params, cv=5) 
forest_cv.fit(X, y)

print("Best score: {}".format(forest_cv.best_score_))
print("Optimal params: {}".format(forest_cv.best_estimator_))

forrest_pred = forest_cv.predict(test)
output=pd.DataFrame({'PassengerId':testids,'Survived':forrest_pred})
output.to_csv('./submission/randomForest.csv', index=False)


##Catboost
import catboost as cb
parameters = {
    'iterations': [5, 10, 15, 20, 25, 50, 100],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [3, 5, 7, 9, 11, 13],
}

model_catboost = cb.CatBoostClassifier(
    verbose=False,
)

model_catboost = GridSearchCV(
    model_catboost, 
    parameters, 
    cv=5,
    scoring='accuracy',
)

model_catboost.fit(X, y)

submission = pd.DataFrame(
    { 
        'PassengerId': testids, 
        'Survived': model_catboost.predict(test).astype(int)
    }
)
submission.to_csv("./submission/submission_catboost.csv", index=False)