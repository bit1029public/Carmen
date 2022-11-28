import dill
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from collections import defaultdict
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import jaccard_score
from sklearn import tree
import os
import time
import argparse
from util import Metrics

import sys
sys.path.append('..')
from util import multi_label_metric

model_name = 'ECC'
resume_path = 'saved/{}/Epoch_49_JA_0.4603_DDI_0.07427.model'.format(model_name)

parser = argparse.ArgumentParser()
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--FT', action='store_true', default=False, help="Fine Tune")
parser.add_argument('--datadir', type=str, default="../data_ordered/", help='dimension')
parser.add_argument('--ftfile', type=str, default="emm", help='finetune file')
parser.add_argument('--cuda', type=int, default=-1, help='use cuda')
parser.add_argument('--epoch', type=int, default=400, help='# of epoches')
parser.add_argument('--early_stop', type=int, default=30, help='early stop number')
parser.add_argument('--resume_path', type=str, default=resume_path, help='resume path')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--seed', type=int, default=1029, help='use cuda')

args = parser.parse_args()

if not os.path.exists(os.path.join("saved", args.model_name)):
    os.makedirs(os.path.join("saved", args.model_name))

def create_dataset(data, diag_voc, pro_voc, med_voc):
    i1_len = len(diag_voc.idx2word)
    i2_len = len(pro_voc.idx2word)
    global output_len
    output_len = len(med_voc.idx2word)
    input_len = i1_len + i2_len
    X = []
    y = []
    for patient in data:
        for visit in patient:
            i1 = visit[0]
            i2 = visit[1]
            o = visit[2]

            multi_hot_input = np.zeros(input_len)
            multi_hot_input[i1] = 1
            multi_hot_input[np.array(i2) + i1_len] = 1

            multi_hot_output = np.zeros(output_len)
            multi_hot_output[o] = 1

            X.append(multi_hot_input)
            y.append(multi_hot_output)

    return np.array(X), np.array(y)

def augment(y_pred, appear_idx):
    m, n = y_pred.shape
    y_pred_aug = np.zeros((m, output_len))
    y_pred_aug[:, appear_idx] = y_pred

    return y_pred_aug

def main():
    # grid_search = False
    data_path = os.path.join(args.datadir, 'records_final_4.pkl')
    voc_path = os.path.join(args.datadir, 'voc_final_4.pkl')
    # data_path = os.path.join(args.datadir, 'records_final.pkl')
    # voc_path = os.path.join(args.datadir, 'voc_final.pkl')

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    metric_obj = Metrics(data, med_voc, args)

    epoch = 100

    np.random.seed(args.seed)
    np.random.shuffle(data)
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_eval = data[split_point+eval_len:]
    data_test = data[split_point:split_point + eval_len]

    train_X, train_y = create_dataset(data_train, diag_voc, pro_voc, med_voc)
    test_X, test_y = create_dataset(data_test, diag_voc, pro_voc, med_voc)
    eval_X, eval_y = create_dataset(data_eval, diag_voc, pro_voc, med_voc)

    """
    some drugs do not appear in the train set (their index is non_appear_idx)
    we omit them during training ECC (resulting in appear_idx)
    and directly not recommend these for test and eval
    """
    # non_appear_idx = np.where(train_y.sum(axis=0) == 0)[0]
    appear_idx = np.where(train_y.sum(axis=0) > 0)[0]
    train_y = train_y[:, appear_idx]

    base_dt = LogisticRegression()

    tic_total_fit = time.time()
    global chains
    chains = [ClassifierChain(base_dt, order='random', random_state=i) for i in range(10)]
    for i, chain in enumerate(chains):
        tic = time.time()
        chain.fit(train_X, train_y)
        fittime = time.time() - tic
        print ('id {}, fitting time: {}'.format(i, fittime))
    print ('total fitting time: {}'.format(time.time() - tic_total_fit))

    # exit()

    tic = time.time()
    y_pred_chains = np.array([augment(chain.predict(test_X), appear_idx) for chain in chains])
    y_prob_chains = np.array([augment(chain.predict_proba(test_X), appear_idx) for chain in chains])
    pretime = time.time() - tic
    print ('inference time: {}'.format(pretime))

    y_pred = y_pred_chains.mean(axis=0)
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    y_prob = y_prob_chains.mean(axis=0)

    metric_obj.set_data(test_y, y_pred, y_prob, save=args.Test)
    ja, prauc, avg_p, avg_r, avg_f1 = metric_obj.run()
    # ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(test_y, y_pred, y_prob)

    # ddi rate
    ddi_adj_path = os.path.join(args.datadir, 'ddi_A_final_4.pkl')
    ddi_A = dill.load(open(ddi_adj_path, 'rb'))
    all_cnt = 0
    dd_cnt = 0
    med_cnt = 0
    visit_cnt = 0
    for adm in y_pred:
        med_code_set = np.where(adm==1)[0]
        visit_cnt += 1
        med_cnt += len(med_code_set)
        for i, med_i in enumerate(med_code_set):
            for j, med_j in enumerate(med_code_set):
                if j <= i:
                    continue
                all_cnt += 1
                if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                    dd_cnt += 1
    ddi_rate = dd_cnt / all_cnt
    print('Epoch: {}, DDI Rate: {:.4}, Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
        epoch, ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, med_cnt / visit_cnt
        ))

if __name__ == '__main__':
    main()   
