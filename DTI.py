import argparse
import os
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, recall_score, f1_score
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForMaskedLM

from model import Model as Modelpre
from model_search import Model as Modelmeta
from pretreatment import cstr_source, cstr_target
from pretreatment import normalize_sym, normalize_row, sparse_mx_to_torch_sparse_tensor
from dataset import loaddata

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.006, help='learning rate')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--alr', type=float, default=3e-4, help='learning rate for architecture parameters')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=50, help='number of epochs for supernet training')
parser.add_argument('--eps', type=float, default=0., help='probability of random sampling')
parser.add_argument('--decay', type=float, default=0.9, help='decay factor for eps')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
prefix = "lr" + str(args.lr) + "_wd" + str(args.wd) + "_h" + str(args.n_hid) + "_alr" + str(args.alr) + \
         "_epoch" + str(args.epochs) + "_cuda" + str(args.gpu) + "_eps" + str(args.eps) + "_d" + str(args.decay)

def main():
    archs = {
            "source": ([[]], [[]]),
            "target": ([[]], [[]])
    }
    torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    datadir = "preprocessed"
    prefix = os.path.join(datadir)

    #* load data
    node_types = np.load(os.path.join(prefix, "node_types.npy"))
    num_node_types = node_types.max() + 1
    node_types = torch.from_numpy(node_types).cuda()

    adjs_offset = pickle.load(open(os.path.join(prefix, "adjs_offset.pkl"), "rb"))
    adjs_pt = []

    for i in range(0,6):
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(adjs_offset[str(i)] + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())
        if(i<4):
            adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(adjs_offset[str(i)].T + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())
    adjs_pt.append(sparse_mx_to_torch_sparse_tensor(sp.eye(adjs_offset['1'].shape[0], dtype=np.float32).tocoo()).cuda())
    adjs_pt.append(torch.sparse.FloatTensor(size=adjs_offset['1'].shape).cuda())

    pos_train, pos_val, pos_test,\
    neg_train, neg_val, neg_test = loaddata(args.seed)

    #* embedding
    print("Start embedding...")
    dg = pd.read_csv('./data/drug_smiles.csv', header=None).values
    pn = pd.read_csv('./data/protein_fasta.csv', header=None).values
    in_dims = []
    node_feats = []
    for k in range(num_node_types):
            in_dims.append((node_types == k).sum().item())
            if (k == 0):
                drug = []
                tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
                model = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
                for param in model.parameters():
                    param.requires_grad_ = None
                for i, d in enumerate(dg):
                    text = d[1]
                    encoded_input = tokenizer(text, return_tensors='pt')
                    with torch.no_grad():
                        output = model(**encoded_input)
                    out = torch.squeeze(output.logits).mean(1)
                    lin = nn.Linear(len(out), 708)
                    out = lin(out).tolist()
                    drug.append(out)
                drug = torch.FloatTensor(drug).cuda()
                node_feats.append(drug)
            if (k == 1):
                pro = []
                tokenizer = AutoTokenizer.from_pretrained("zjukg/OntoProtein")
                model = AutoModelForMaskedLM.from_pretrained("zjukg/OntoProtein")
                for param in model.parameters():
                    param.requires_grad_ = None
                for i, p in enumerate(pn):
                    text = p[1]
                    encoded_input = tokenizer(text, return_tensors='pt')
                    with torch.no_grad():
                        output = model(**encoded_input)
                    out = torch.squeeze(output.logits).mean(1)
                    lin = nn.Linear(len(out), 1512)
                    out = lin(out).tolist()
                    pro.append(out)
                pro = torch.FloatTensor(pro).cuda()
                node_feats.append(pro)
            if (k > 1):
                i = torch.stack((torch.arange(in_dims[-1], dtype=torch.long), torch.arange(in_dims[-1], dtype=torch.long)))
                v = torch.ones(in_dims[-1])
                node_feats.append(torch.sparse.FloatTensor(i, v, torch.Size([in_dims[-1], in_dims[-1]])).cuda())
    assert (len(in_dims) == len(node_feats))
    print("Embedding over!")
    print("Start train AMG...")

    meta_auc_best = None
    t = [4]
    for i in range(10):
        # 开始训练
        modelmeta_s = Modelmeta(in_dims, args.n_hid, len(adjs_pt), t, cstr_source['Drug']).cuda()
        modelmeta_t = Modelmeta(in_dims, args.n_hid, len(adjs_pt), t, cstr_target['Drug']).cuda()
        optimizer_w = torch.optim.Adam(
            list(modelmeta_s.parameters()) + list(modelmeta_t.parameters()),
            lr=args.lr,
            weight_decay=args.wd
        )
        optimizer_a = torch.optim.Adam(
            modelmeta_s.alphas() + modelmeta_t.alphas(),
            lr=args.alr
        )
        eps = args.eps

        for epoch in range(args.epochs):
            train_error, val_error, auc, aupr = train_meta(node_feats, node_types, adjs_pt, pos_train[i],neg_train[i], pos_val[i],
                                                      neg_val[i], modelmeta_s, modelmeta_t, optimizer_w, optimizer_a, eps,pos_test[i], neg_test[i])
            if(meta_auc_best == None or meta_auc_best < auc):
                meta_auc_best = auc
                archs["source"] = modelmeta_s.parse()
                archs["target"] = modelmeta_t.parse()
            eps = eps * args.decay
    print("Train AMG over!")

    avg_auc1 = []
    avg_aupr1 = []
    for i in tqdm(range(10)):
        steps_s = [len(meta) for meta in archs["source"][0]]
        steps_t = [len(meta) for meta in archs["target"][0]]
        modelpre_s = Modelpre(in_dims, args.n_hid, steps_s, dropout=0.3).cuda()
        modelpre_t = Modelpre(in_dims, args.n_hid, steps_t, dropout=0.3).cuda()
        optimizer = torch.optim.Adam(
            list(modelpre_s.parameters()) + list(modelpre_t.parameters()),
            lr=0.006,
            weight_decay=0.09
        )
        auc_best = None
        aupr_best = None
        for epoch in range(args.epochs):
            train_loss = train_pre(node_feats, node_types, adjs_pt, pos_train[i], neg_train[i], modelpre_s, modelpre_t, optimizer, archs["source"], archs["target"])
            auc, aupr = infer(node_feats, node_types, adjs_pt, pos_val[i], neg_val[i], pos_test[i],neg_test[i], modelpre_s, modelpre_t, archs["source"],archs["target"])
            if auc_best is None or auc > auc_best:
                auc_best = auc
                aupr_best = aupr

        avg_auc1.append(auc_best)
        avg_aupr1.append(aupr_best)
    print("AVG_AUC {}; AVG_AUPR {}".format(np.mean(avg_auc1), np.mean(avg_aupr1)))



def train_meta(node_feats, node_types, adjs, pos_train, neg_train, pos_val, neg_val, model_s, model_t, optimizer_w, optimizer_a, eps,pos_test, neg_test):

    idxes_seq_s, idxes_res_s = model_s.sample(eps)
    idxes_seq_t, idxes_res_t = model_t.sample(eps)

    optimizer_w.zero_grad()
    out_s = model_s(node_feats, node_types, adjs, idxes_seq_s, idxes_res_s)
    out_t = model_t(node_feats, node_types, adjs, idxes_seq_t, idxes_res_t)
    loss_w = - torch.mean(F.logsigmoid(torch.mul(out_s[pos_train[:, 0]], out_t[pos_train[:, 1]]).sum(dim=-1)) + \
                        F.logsigmoid(- torch.mul(out_s[neg_train[:, 0]], out_t[neg_train[:, 1]]).sum(dim=-1)))
    loss_w.backward()
    optimizer_w.step()

    optimizer_a.zero_grad()
    out_s = model_s(node_feats, node_types, adjs, idxes_seq_s, idxes_res_s)
    out_t = model_t(node_feats, node_types, adjs, idxes_seq_t, idxes_res_t)
    loss_a = - torch.mean(F.logsigmoid(torch.mul(out_s[pos_val[:, 0]], out_t[pos_val[:, 1]]).sum(dim=-1)) + \
                        F.logsigmoid(- torch.mul(out_s[neg_val[:, 0]], out_t[neg_val[:, 1]]).sum(dim=-1)))
    loss_a.backward()
    optimizer_a.step()

    pos_test_prod = torch.mul(out_s[pos_test[:, 0]], out_t[pos_test[:, 1]]).sum(dim=-1)
    neg_test_prod = torch.mul(out_s[neg_test[:, 0]], out_t[neg_test[:, 1]]).sum(dim=-1)
    y_true_test = np.zeros((pos_test.shape[0] + neg_test.shape[0]), dtype=np.int64)
    y_true_test[:pos_test.shape[0]] = 1
    y_pred_test = np.concatenate(
        (torch.sigmoid(pos_test_prod).cpu().detach().numpy(), torch.sigmoid(neg_test_prod).cpu().detach().numpy()))

    auc = roc_auc_score(y_true_test, y_pred_test)
    aupr = average_precision_score(y_true_test, y_pred_test)

    return loss_w.item(), loss_a.item(),auc,aupr

def train_pre(node_feats, node_types, adjs, pos_train, neg_train, model_s, model_t, optimizer,s,t):
    model_s.train()
    model_t.train()
    optimizer.zero_grad()
    out_s = model_s(node_feats, node_types, adjs, s[0], s[1])
    out_t = model_t(node_feats, node_types, adjs, t[0], t[1])
    loss = - torch.mean(F.logsigmoid(torch.mul(out_s[pos_train[:, 0]], out_t[pos_train[:, 1]]).sum(dim=-1)) + \
                        F.logsigmoid(- torch.mul(out_s[neg_train[:, 0]], out_t[neg_train[:, 1]]).sum(dim=-1)))
    loss.backward()
    optimizer.step()
    return loss.item()

def infer(node_feats, node_types, adjs, pos_val, neg_val, pos_test, neg_test, model_s, model_t,s,t):
    model_s.eval()
    model_t.eval()
    with torch.no_grad():
        out_s = model_s(node_feats, node_types, adjs, s[0], s[1])
        out_t = model_t(node_feats, node_types, adjs, t[0], t[1])

    pos_val_prod = torch.mul(out_s[pos_val[:, 0]], out_t[pos_val[:, 1]]).sum(dim=-1)
    neg_val_prod = torch.mul(out_s[neg_val[:, 0]], out_t[neg_val[:, 1]]).sum(dim=-1)
    loss = - torch.mean(F.logsigmoid(pos_val_prod) + F.logsigmoid(- neg_val_prod))

    pos_test_prod = torch.mul(out_s[pos_test[:, 0]], out_t[pos_test[:, 1]]).sum(dim=-1)
    neg_test_prod = torch.mul(out_s[neg_test[:, 0]], out_t[neg_test[:, 1]]).sum(dim=-1)

    y_true_test = np.zeros((pos_test.shape[0] + neg_test.shape[0]), dtype=np.int64)
    y_true_test[:pos_test.shape[0]] = 1
    y_pred_test = np.concatenate(
        (torch.sigmoid(pos_test_prod).cpu().numpy(), torch.sigmoid(neg_test_prod).cpu().numpy()))

    auc = roc_auc_score(y_true_test, y_pred_test)
    aupr = average_precision_score(y_true_test, y_pred_test)

    return auc, aupr

if __name__ == '__main__':
    main()