from LDAS.ldas import LDAS
from LDAS.utils import load_norm_data


if __name__ == '__main__':
    path = r'dataSet_name'
    X, y, Maj_num, Min_num, IR, features = load_norm_data(path)
    X_re, y_re = LDAS().fit_sample(X, y)


