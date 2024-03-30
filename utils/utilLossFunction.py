import numpy as np

class utilLossFunction:
    def mean_squared_error(y_pred, y_tr):
        sum = 0
        for i in range(len(y_tr)):
            sum += (y_pred - y_tr) ** 2
        return sum / len(y_tr)

    def softmax(values):
        exp_values = np.exp(values)
        exp_values_sum = np.sum(exp_values)
        return exp_values / exp_values_sum

    def cross_entropy(y_pred,y_true):
        y_pred = utilLossFunction.softmax(y_pred)

        loss = 0
        for i in range(len(y_pred)):
            loss += y_true[i]*np.log(y_pred[i])*-1
        return loss

