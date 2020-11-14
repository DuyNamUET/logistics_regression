import numpy as np
import pandas as pd

def f1_statistic(Y_truth, Y_predict):
    Y_set = list(set(Y_truth))
    length = len(Y_set)
    print(length)
    f1 = np.zeros([length, length])
    print(f1.size)
    for i in range(len(Y_truth)):
        f1[Y_predict[i], Y_truth[i]] += 1
    return f1

if __name__ == "__main__":
    df = pd.read_csv("mobile_price/train.csv")
    Y_truth = df[df.columns[-1]].values
    # print(Y_truth)
    Y_predict = df[df.columns[-1]].values[::-1]
    # print(Y_predict)
    f1 = f1_statistic(Y_truth, Y_predict)
    # print(f1)

    test = np.array([[4,6,3],[1,2,0],[1,2,6]])
    print(test)
    print(f1_value(test))