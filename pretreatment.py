import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import os
import sys
import pickle

from sklearn.model_selection import KFold

cstr_source = {
    "Drug": [0]
}

cstr_target = {
    "Drug": [1]
}


def normalize_sym(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_row(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx.tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def pretreatment_Drug(prefix):
    dp = pd.read_csv(os.path.join(prefix, "drug_target.dat"), encoding='utf-8', delimiter=',',
                     names=['did', 'pid', 'rating']).reset_index(drop=True)
    dd = pd.read_csv(os.path.join(prefix, "drug_drug.dat"), encoding='utf-8', delimiter=',',
                     names=['d1', 'd2', 'weight']).reset_index(drop=True)
    pp = pd.read_csv(os.path.join(prefix, "pro_pro.dat"), encoding='utf-8', delimiter=',',
                     names=['p1', 'p2', 'weight']).reset_index(drop=True)
    de = pd.read_csv(os.path.join(prefix, "drug_dis.dat"), encoding='utf-8', delimiter=',',
                     names=['did', 'dis', 'weight']).reset_index(drop=True)
    pe = pd.read_csv(os.path.join(prefix, "protein_dis.dat"), encoding='utf-8', delimiter=',',
                     names=['p1', 'dis', 'weight']).reset_index(drop=True)
    ds = pd.read_csv(os.path.join(prefix, "drug_se.dat"), encoding='utf-8', delimiter=',',
                     names=['d1', 'se', 'weight']).reset_index(drop=True)

    offsets = {'p': 708, 'd': 708 + 1512}
    offsets['e'] = offsets['d'] + 5603
    offsets['s'] = offsets['e'] + 4192
    print(offsets['s'])
    # * node types
    node_types = np.zeros((offsets['s'],), dtype=np.int32)
    node_types[offsets['p']:offsets['d']] = 1
    node_types[offsets['d']:offsets['e']] = 2
    node_types[offsets['e']:] = 3

    if not os.path.exists("./preprocessed/node_types.npy"):
        np.save("./preprocessed/node_types", node_types)

    dp_pos = dp[dp['rating'] == 1].to_numpy()[:, :2]
    dp_pos[:, 1] += offsets['p']
    neg_ratings = dp[dp['rating'] == 0].to_numpy()[:, :2]
    assert (dp_pos.shape[0] + neg_ratings.shape[0] == dp.shape[0])
    neg_ratings[:, 1] += offsets['p']
    np.save("./preprocessed/pos_ratings_offset", dp_pos)
    np.save("./preprocessed/neg_ratings_offset", neg_ratings)

    adjs_offset = {}

    ## dp
    dp_npy = dp_pos[np.arange(dp_pos.shape[0])]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[dp_npy[:, 0], dp_npy[:, 1] + offsets['p']] = 1
    adjs_offset['0'] = sp.coo_matrix(adj_offset)
    print(len(dp_npy))

    # de
    de_npy = de.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[de_npy[:, 0], de_npy[:, 1] + offsets['d']] = 1
    adjs_offset['1'] = sp.coo_matrix(adj_offset)
    print(len(de_npy))

    # pe
    pe_npy = pe.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[pe_npy[:, 0] + offsets['p'], pe_npy[:, 1] + offsets['d']] = 1
    adjs_offset['2'] = sp.coo_matrix(adj_offset)
    print(len(pe_npy))

    # ds
    ds_npy = ds.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[ds_npy[:, 0], ds_npy[:, 1] + offsets['e']] = 1
    adjs_offset['3'] = sp.coo_matrix(adj_offset)
    print(len(ds_npy))

    # dd
    dd_npy = dd.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[dd_npy[:, 0], dd_npy[:, 1]] = 1
    adjs_offset['4'] = sp.coo_matrix(adj_offset)
    print(len(dd_npy))

    # pp
    pp_npy = pp.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[pp_npy[:, 0] + offsets['p'], pp_npy[:, 1] + offsets['p']] = 1
    adjs_offset['5'] = sp.coo_matrix(adj_offset)
    print(len(pp_npy))

    f2 = open("./preprocessed/adjs_offset.pkl", "wb")
    pickle.dump(adjs_offset, f2)
    f2.close()

if __name__ == '__main__':
    prefix = "./data"
    pretreatment_Drug(prefix)