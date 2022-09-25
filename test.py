# this file is for testing only
from random import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt


data = pd.read_csv('gamingData5.csv', delimiter=';')

# process data on Q8 to two classes
data.dropna(how='any', axis=0, inplace=True)
data.loc[data['Q8'] < 2, 'Q8'] = 0
data.loc[data['Q8'] >= 2, 'Q8'] = 1

X = data[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']]
X_sum = X.iloc[:, 0]
for i in range(1, X.shape[1]):
    X_sum += X.iloc[:, i]
y = data['Q8']

X_train, X_test, y_train, y_test = train_test_split(X_sum, y, test_size=0.3, random_state=0)
log_regression = LogisticRegression()
log_regression.fit(X_train.values.reshape(-1,1), y_train)
y_pred_proba = log_regression.predict_proba(X_test.values.reshape(-1,1))[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr, tpr, label="AUC="+str(auc))
plt.ylabel("TPR")
plt.xlabel("FPR")
plt.legend(loc=4)
plt.show()


parameters = [2, 2, 2, 2, 2, 2, 2]
# X = X.copy()

# for i in range(len(parameters)):
#     X.iloc[:,i] *= parameters[i]



#define the objective function:
# def obj_fun(parameters, *data):
#     data = np.asarray(data) # conver data into np array
#     # Using logistic regression to re-scale data within [0,1]
#     result = sigmoid(np.dot(data[:, :7], para_transposed))
#     # assign new data with 1 when >= 0.5, 0 otherwise
#     diff = np.subtract(data[:, 7], result)
#     diff = diff[~np.isnan(diff)]
#     diff_squ = np.square(diff)
#     diff_squ_sum = np.sum(diff_squ) 
#     # y_test = data[:, 7]
#     # y_pred = result 
#     # auc = roc_auc_score(y_test, y_pred)

#     return diff_squ_sum



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