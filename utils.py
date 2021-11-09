import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import Counter


def load_norm_data(path):
    df = pd.read_csv(path, header=None)
    data = df.values
    label = data[:, -1]
    columns = data.shape[1]
    x = data[:, :columns - 1]

    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)

    Maj_num = Counter(label)[1]
    Min_num = Counter(label)[0]
    IR = Maj_num / Min_num

    print("Instances: {0} ,Features: {1} ,Maj: {2} ,Min: {3} ,IR: {4} ".format(len(label), columns - 1, Maj_num,
                                                                               Min_num,
                                                                               round(IR, 2)))
    return x, label, Maj_num, Min_num, round(IR, 2), columns - 1


def add_label(X, y):
    data = np.insert(X, X.shape[1], y, axis=1)
    return data

