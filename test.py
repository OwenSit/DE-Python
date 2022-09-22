# this file is for testing only
from random import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt


data = pd.read_csv('gamingData5.csv', delimiter=';')
# for i, x in enumerate(data['Q8']):
#     if x >= 2:
#         data['Q8'][i] = 1
#     else:
#         data['Q8'][i] = 0

# process data on Q8 to two classes
data.dropna(how='any', axis=0, inplace=True)
data.loc[data['Q8'] < 2, 'Q8'] = 0
data.loc[data['Q8'] >= 2, 'Q8'] = 1

X = data[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']]
y = data['Q8']

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_regression = LogisticRegression()
log_regression.fit(X_train, y_train)
y_pred_proba = log_regression.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr, tpr, label="AUC="+str(auc))
plt.ylabel("TPR")
plt.xlabel("FPR")
plt.legend(loc=4)
plt.show()
# url = "https://raw.githubusercontent.com/Statology/Python-Guides/main/default.csv"
# data = pd.read_csv(url)

# X = data[['student']]
# y = data['default']
# # print(y.to_numpy())

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# log_regression = LogisticRegression()

# log_regression.fit(X_train, y_train)


# y_pred_proba = log_regression.predict_proba(X_test)[::,1]


# fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
# auc = metrics.roc_auc_score(y_test, y_pred_proba)

# plt.plot(fpr,tpr,label="AUC="+str(auc))
# plt.ylabel("TPR")
# plt.xlabel("FPR")
# plt.legend(loc=4)
# plt.show()