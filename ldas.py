import numpy as np
from LDAS.utils import add_label
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import random
from sklearn import preprocessing


class LDAS:
    def __init__(self, k=5, k1=5, w=0.02):

        assert k in [3, 5, 7, 9, 11]
        assert k1 in [3, 5, 7, 9, 11]
        assert w in [0.01, 0.02, 0.03, 0.05, 0.1]

        self.k = k
        self.k1 = k1
        self.w = w

        self.nnarray = []
        self.Synthetic = []
        self.maj_index = []
        self.min_index = []

    def set_dc(self, N, dist):
        K = int(np.ceil(self.w * N))
        dc = np.average(dist[:, K])
        return dc

    def cal_density(self, N, min_sample):
        neigh = NearestNeighbors(n_neighbors=N).fit(min_sample)
        dist, indices = neigh.kneighbors(min_sample)
        self.nnarray = indices
        dc = self.set_dc(N, dist)
        p = np.zeros(N)

        for i in range(N):
            for j in range(1, N):
                p[i] += np.exp(-(dist[i, j] ** 2 / dc ** 2))
        return self.normalize(p)

    def remove_Overlap_Majority(self, X, y, indices, dist, density, N):
        q = {}
        p = np.zeros(N)

        del_index_list = []
        for i, index in enumerate(self.min_index[0]):

            neigh_label = y[indices[index, 1:self.k1 + 1]]

            K_Nmaj = Counter(neigh_label)[1]
            p[i] = K_Nmaj / self.k1

            neigh_maj_index = np.where(neigh_label == 1)[0] + 1

            for j in neigh_maj_index:

                dist_to_min = dist[index, j]

                if dist_to_min == 0:
                    if indices[index, j] not in del_index_list:
                        del_index_list.append(indices[index, j])
                else:
                    overlap_value = density[i] / dist_to_min
                    if indices[index, j] in q:
                        q[indices[index, j]] += overlap_value
                    else:
                        q[indices[index, j]] = overlap_value

        ol_list = list(q.values())
        Thre = np.mean(ol_list)

        for key, value in q.items():
            if value >= Thre:
                del_index_list.append(key)

        X_y = add_label(X, y)
        X_y_removed = np.delete(X_y, del_index_list, axis=0)
        X_removed, y_removed = X_y_removed[:, :-1], X_y_removed[:, -1]
        maj_temp = np.where(y_removed == 1)[0]

        need_num = len(maj_temp) - N

        return X_removed, y_removed, need_num, p

    def cal_weight(self, a, b):
        return self.normalize(a ** 2 + b ** 2)

    def cal_border_degree(self, p_list):
        border_degree_list = []
        for p in p_list:
            if p == 1 or p == 0:
                border_degree = 0
            else:
                border_degree = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            border_degree_list.append(border_degree)
        return border_degree_list

    def normalize(self, a):
        a = a.reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        a = min_max_scaler.fit_transform(a)
        return a.reshape(1, -1)[0]

    def cal_num_to_gen(self, weight, G):
        sum = np.sum(weight)
        return np.rint((weight / sum) * G)

    def populate(self, i, g, p):
        N = len(p)
        if N < self.k + 1:
            range_k = N - 1
        else:
            range_k = self.k
        n = self.nnarray[i, :range_k + 1]

        numattrs = p.shape[1]
        count = 0
        while count < g[i]:
            s = np.zeros(numattrs)
            nn = random.randint(1, range_k)

            for atti in range(numattrs):
                gap = random.random()
                dif = p[n[nn], atti] - p[i, atti]
                s[atti] = p[i, atti] + gap * dif
            self.Synthetic.append(s)

            count += 1

    def fit_sample(self, X, y):
        self.min_index = np.where(y == 0)
        self.maj_index = np.where(y == 1)

        min_sample = X[self.min_index]
        maj_sample = X[self.maj_index]

        N = len(min_sample)
        M = len(maj_sample)

        density = self.cal_density(N, min_sample)
        neigh = NearestNeighbors(n_neighbors=self.k1 + 1).fit(X)
        dist, indices = neigh.kneighbors(X)

        X_removed, y_removed, G, p = self.remove_Overlap_Majority(X, y, indices, dist, density, N)

        border_degree_list = self.cal_border_degree(p)
        border_degree_list = np.array(border_degree_list)

        if G <= 0:
            return X_removed, y_removed
        else:

            weight = self.cal_weight(density, border_degree_list)

            g = np.array(self.cal_num_to_gen(weight, G), dtype=int)

            # --------------------------Oversampling----------------------------------
            for i in range(N):
                self.populate(i, g, min_sample)
            self.Synthetic = np.array(self.Synthetic)
            # --------------------------Oversampling----------------------------------

            if len(self.Synthetic) > 0:
                S = np.concatenate((add_label(X_removed, y_removed), add_label(self.Synthetic, 0)), axis=0)
                X_new = S[:, :-1]
                y_new = S[:, -1]
                return X_new, y_new
            else:
                return X_removed, y_removed
