import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import roc_auc_score

np.set_printoptions(threshold=np.inf)


def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig


#define the objective function:
def obj_fun(parameters, data):
    para_np_arr = np.array(parameters) # convert the input data into np array
    print(para_np_arr)
    para_transposed = np.transpose(para_np_arr)
    data = np.asarray(data) 
    # Using logistic regression to re-scale data within [0,1]
    result = sigmoid(np.dot(data[:, :7], para_transposed))
    # assign new data with 1 when >= 0.5, 0 otherwise
    result = result >= 0.5
    y_test = data[:, 7]
    y_pred = result 
    auc = roc_auc_score(y_test, y_pred)

    return -1*auc


#in the data, women='0', man='1',children='0', students='1', parents='2'
input = np.genfromtxt('gamingData5.csv', skip_header=1, delimiter=';')
data = input[:, :8]
#modify data in the output (8th) column
for row in range(len(data)):
    if data[row][7] >= 2:
        data[row][7] = 1
    else:
        data[row][7] = 0

bounds = [(0.01, 3), (0.01, 3), (0.01, 3), (0.01, 3), (0.01, 3), (0.01, 3),
          (0.01, 3)]

parameters = [2,2,2,2,2,2,2]
# result = differential_evolution(obj_fun, bounds, args=data)
print(obj_fun(parameters, data))