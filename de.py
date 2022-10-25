import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt


def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig


# define the objective function:
def obj_fun(parameters, *data):
    para_np_arr = np.array(parameters)  # convert the input data into np array
    para_transposed = np.transpose(para_np_arr)
    data = np.asarray(data)  # conver data into np array
    # Using logistic regression to re-scale data within [0,1]
    result = np.dot(data[:, :7], para_transposed)
    # set binalize of Q8 based on value 2.5
    # classification with the threshold set to 17.5 as 7 * 2.5 = 17.5
    # result[result >= 17.5] = 1
    # result[result < 17.5] = 0
    for i in range(result.shape[0]):
        if (result[i]) >= 10.5:
            result[i] = 1
        else:
            result[i] = 0
    # assign new data with 1 when >= 0.5, 0 otherwise
    diff = np.subtract(data[:, 7], result)
    diff = diff[~np.isnan(diff)]
    diff_squ = np.square(diff)
    diff_squ_sum = np.sum(diff_squ)
    # y_test = data[:, 7]
    # y_pred = result
    # auc = roc_auc_score(y_test, y_pred)
    # print(diff_squ_sum)
    return -1*diff_squ_sum


input = np.genfromtxt('woman_data_cleaned.csv', skip_header=1, delimiter=',')
data = input[:, :8]
Q8_THRESHOLD = 1.5
Q17_THRESHOLD = 7 * Q8_THRESHOLD
# modify data in the output (8th) column
for row in range(len(data)):
    if data[row][7] >= 1.5:
        data[row][7] = 1
    else:
        data[row][7] = 0

bounds = [(0.01, 3), (0.01, 3), (0.01, 3), (0.01, 3), (0.01, 3), (0.01, 3),
          (0.01, 3)]

# when all weights are equal to 1
data = np.asarray(data)
pre_result = data[:, :7].sum(axis=0)

result = differential_evolution(obj_fun, bounds, args=data)
# print(obj_fun(parameters, data))
# print(result.x)

# retrive the ROC plot of the DE result
para = np.array(result.x)
para_transposed = np.transpose(para)
result_dot = np.dot(data[:, :7], para_transposed)
for i in range(result_dot.shape[0]):
    if (result_dot[i]) >= 10.5:
        result_dot[i] = 1
    else:
        result_dot[i] = 0

fpr, tpr, _ = metrics.roc_curve(data[:, 7],  result_dot)
auc = metrics.roc_auc_score(data[:, 7], result_dot)
plt.plot(fpr, tpr, label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()
