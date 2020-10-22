#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 21:37:21 2020

@author: yangzhang
"""

#############################
#### logistic regression ####
#############################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

logR = LogisticRegression()
logR.fit(train_X,train_y)
y_pred = logR.predict(val_X)

# printing confision matrix
pd.DataFrame(confusion_matrix(val_y,y_pred),\
            columns=["Predicted Not-Survived", "Predicted Survived"],\
            index=["Not-Survived","Survived"] )
    
    
from sklearn.metrics import accuracy_score
accuracy_score(val_y, y_pred)

test_preds = logR.predict(X_test_scale)

#Submission
output = pd.DataFrame({'PassengerId': testids,
                       'Survived': test_preds})
output.to_csv('submission_logisticR.csv', index=False)