import numpy as np
from sklearn.model_selection import KFold

def loaddata(data_seed):
        np.random.seed(data_seed)
        pos_offset = np.load("./preprocessed/pos_ratings_offset.npy")
        neg_offset = np.load("./preprocessed/neg_ratings_offset.npy")
        rs = np.random.randint(0, 1000, 1)[0]
        kf = KFold(n_splits=10, shuffle=True, random_state=rs)

        pos_train = []
        pos_val = []
        pos_test = []
        indices1 = np.zeros(shape=(pos_offset.shape[0]*10,), dtype=int)
        count = 0
        for i in range(pos_offset.shape[0]):
            for j in range(10):
                indices1[count] = i
                count = count + 1
        pos_offset = pos_offset[indices1]

        for train_idx, test in kf.split(indices1):
            train_n, val = np.array_split(train_idx, [int(len(train_idx) * 0.8)])
            pos_train.append(pos_offset[train_n])
            pos_val.append(pos_offset[val])
            pos_test.append(pos_offset[test])

        neg_train = []
        neg_val = []
        neg_test = []
        indices2 = np.arange(neg_offset.shape[0])
        np.random.shuffle(indices2)
        indices2 = indices2[:pos_offset.shape[0]]
        neg_offset = neg_offset[indices2]
        for train_idx, test in kf.split(indices2):
            train_n, val = np.array_split(train_idx, [int(len(train_idx) * 0.8)])
            neg_train.append(neg_offset[train_n])
            neg_val.append(neg_offset[val])
            neg_test.append(neg_offset[test])
        return pos_train, pos_val, pos_test, neg_train, neg_val, neg_test