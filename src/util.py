from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os
import traceback
import pdb
import warnings
import dill
from collections import Counter
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from collections import defaultdict, namedtuple
import torch
warnings.filterwarnings('ignore')

Args = namedtuple("Args", ['datadir'])
# cert_words = ['0054-3194', '0517-1010', '66213-423', '66213-425', '0225-0295', '66213-423', '66213-425']
cert_words = ['0456-4020', '54838-540', '60505-2519', '0456-2010', '51079-543']

def _get_metric(datadir, modelname):
    args = Args(datadir=datadir)
    if args.MIMIC == 4:
        data_path = os.path.join(args.datadir, 'records_final_4.pkl')
        voc_path = os.path.join(args.datadir, 'voc_final_4.pkl')
    else:
        data_path = os.path.join(args.datadir, 'records_final.pkl')
        voc_path = os.path.join(args.datadir, 'voc_final.pkl')        
    voc = dill.load(open(voc_path, 'rb'))
    med_voc = voc['med_voc']
    data = dill.load(open(data_path, 'rb'))

    main_met_obj = Metrics(data, med_voc, args)
    loadpath = os.path.join("saved", modelname, 'test_gt_pred_prob.pkl')
    gt, pred, prob = dill.load(open(loadpath, 'rb'))
    main_met_obj.set_data(gt=gt, pred=pred, prob=prob)
    return main_met_obj

def get_ehr_adj(records, Nmed, no_weight=True, filter_th=None) -> np.array:
    ehr_adj = np.zeros((Nmed, Nmed))
    for patient in records:
        for adm in patient:
            med_set = adm[2]
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j<=i:
                        continue
                    ehr_adj[med_i, med_j] += 1
                    ehr_adj[med_j, med_i] += 1

    if filter_th is not None:
        ehr_adj[ehr_adj <= filter_th] = 0
    if no_weight:
        ehr_adj = ehr_adj.astype(bool).astype(int)

    return ehr_adj

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

# use the same metric from DMNC
def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def transform_split(X, Y):
    x_train, x_eval, y_train, y_eval = train_test_split(X, Y, train_size=2/3, random_state=1203)
    x_eval, x_test, y_eval, y_test = train_test_split(x_eval, y_eval, test_size=0.5, random_state=1203)
    return x_train, x_eval, x_test, y_train, y_eval, y_test

def sequence_output_process(output_logits, filter_token):
    pind = np.argsort(output_logits, axis=-1)[:, ::-1]

    out_list = []
    break_flag = False
    for i in range(len(pind)):
        if break_flag:
            break
        for j in range(pind.shape[1]):
            label = pind[i][j]
            if label in filter_token:
                break_flag = True
                break
            if label not in out_list:
                out_list.append(label)
                break
    y_pred_prob_tmp = []
    for idx, item in enumerate(out_list):
        y_pred_prob_tmp.append(output_logits[idx, item])
    sorted_predict = [x for _, x in sorted(zip(y_pred_prob_tmp, out_list), reverse=True)]
    return out_list, sorted_predict


def sequence_metric(y_gt, y_pred, y_prob, y_label):
    def average_prc(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b]==1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score


    def average_recall(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score


    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if (average_prc[idx] + average_recall[idx]) == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score


    def jaccard(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_pred_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_pred_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob_label, k):
        precision = 0
        for i in range(len(y_gt)):
            TP = 0
            for j in y_prob_label[i][:k]:
                if y_gt[i, j] == 1:
                    TP += 1
            precision += TP / k
        return precision / len(y_gt)
    try:
        auc = roc_auc(y_gt, y_prob)
    except ValueError:
        auc = 0
    p_1 = precision_at_k(y_gt, y_label, k=1)
    p_3 = precision_at_k(y_gt, y_label, k=3)
    p_5 = precision_at_k(y_gt, y_label, k=5)
    f1 = f1(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_label)
    avg_prc = average_prc(y_gt, y_label)
    avg_recall = average_recall(y_gt, y_label)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)


def multi_label_metric(y_gt, y_pred, y_prob):

    def jaccard(y_gt, y_pred):
        score = []
        # pdb.set_trace()
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if len(union) == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)

    # roc_auc
    try:
        auc = roc_auc(y_gt, y_prob)
    except:
        auc = 0
    # precision
    p_1 = precision_at_k(y_gt, y_prob, k=1)
    p_3 = precision_at_k(y_gt, y_prob, k=3)
    p_5 = precision_at_k(y_gt, y_prob, k=5)
    # macro f1
    f1 = f1(y_gt, y_pred)
    # precision
    prauc = precision_auc(y_gt, y_prob)
    # jaccard
    ja = jaccard(y_gt, y_pred)
    # pre, recall, f1
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)

def dangerous_pair_num(record, t_record, path, ehr_path):
    # ddi rate
    if isinstance(path, str):
        ddi_A = dill.load(open(path, 'rb'))
    else:
        ddi_A = path
    if isinstance(ehr_path, str):
        ehr_A = dill.load(open(path, 'rb'))
    else:
        ehr_A = ehr_path
    all_cnt = 0
    dd_cnt = 0
    test_ddi_A = np.zeros_like(ddi_A)
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        # print((med_i, med_j))
                        # dd_cnt += 1
                        test_ddi_A[med_i, med_j] = 1
                        test_ddi_A[med_j, med_i] = 1
                        # if ehr_A[med_i, med_j] == 1 or ehr_A[med_j, med_i] == 1:
                            # dd_cnt -= 1
    for patient in t_record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    # all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        # print((med_i, med_j))
                        test_ddi_A[med_i, med_j] = 0
                        test_ddi_A[med_j, med_i] = 0                        
                        # dd_cnt += 1   
    if all_cnt == 0:
        return 0
    # return dd_cnt / all_cnt
    # return np.sum(test_ddi_A) / all_cnt
    return np.sum(test_ddi_A) # the number of the adverse DDI pairs
    # return dd_cnt

def ddi_rate_score(record, path):
    # ddi rate
    if isinstance(path, str):
        ddi_A = dill.load(open(path, 'rb'))
    else:
        ddi_A = path
    all_cnt = 0
    dd_cnt = 0
    # test_ddi_A = np.zeros_like(ddi_A)
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        # print((med_i, med_j))
                        dd_cnt += 1
                        # test_ddi_A[med_i, med_j] = 1
                        # test_ddi_A[med_j, med_i] = 1
                        # if ehr_A[med_i, med_j] == 1 or ehr_A[med_j, med_i] == 1:
                            # dd_cnt -= 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt
    # return np.sum(test_ddi_A) / all_cnt
    # return np.sum(test_ddi_A)
    # return dd_cnt

class Metrics:
    def __init__(self, data, med_voc, args=None):
        self.med_voc = med_voc
        data = data
        cnts = Counter()
        self.args = args
        for p in data:
            for v in p:
                meds = v[-1]
                cnts = cnts + Counter(meds)
        # divid the many, medium, few
        sorted_cnt = sorted(list(cnts.items()), key=lambda x: (-x[1], x[0]))
        self.medidx_ordered_desc, self.freqs_desc = list(zip(*sorted_cnt))
        Max_freq = sorted_cnt[0][1]
        th1, th2 = 0.6, 0.2
        th1, th2 = Max_freq*th1, Max_freq*th2
        self.stacks = defaultdict(list)
        i = 0
        while i < len(self.freqs_desc):
            if self.freqs_desc[i] <= 1000:
                break
            i += 1
        print("thre idx: {}".format(i))
        self.lowfreqmedidx = self.medidx_ordered_desc[0:30]

        if args.MIMIC == 4:
            ddi_adj_path = os.path.join(args.datadir, 'ddi_A_final_4.pkl')
            ehr_path = os.path.join(args.datadir, 'ehr_adj_final_4.pkl')
        else:
            ddi_adj_path = os.path.join(args.datadir, 'ddi_A_final.pkl')
            ehr_path = os.path.join(args.datadir, 'ehr_adj_final.pkl')            
        # ddi_adj_path = os.path.join(args.datadir, 'ddi_A_final_mod.pkl')

        self.ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
        self.ehr_adj = dill.load(open(ehr_path, 'rb'))
    
    def feed_data(self, y_gt, y_pred, y_prob):
        """
        patient level: (N_v, N_med)
        """
        for v_idx in range(len(y_gt)):
            cur_gt = y_gt[v_idx]
            cur_pred = y_pred[v_idx]
            cur_prob = y_prob[v_idx]
            self.stacks['gt'].append(cur_gt)
            self.stacks['pred'].append(cur_pred)
            self.stacks['prob'].append(cur_prob)

    def set_data(self, gt=None, pred=None, prob=None, save=False):
        if gt is None:
            self.gt = np.stack(self.stacks['gt'])
            self.pred = np.stack(self.stacks['pred'])
            self.prob = np.stack(self.stacks['prob'])
            self.stacks = defaultdict(list)
        else:
            self.gt, self.pred, self.prob = gt, pred, prob
        if save:
            model_name =self.args.model_name
            savefile = os.path.join('saved', model_name, 'test_gt_pred_prob.pkl')
            dill.dump((self.gt, self.pred, self.prob), open(savefile, 'wb'))
        return

    def get_metric_res(self):
        ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(
            self.gt, self.pred, self.prob)

        return ja, prauc, avg_p, avg_r, avg_f1
    
    def get_metric_res_for_cert_meds(self, meds:list, is_idx=True):
        meds = list(set(meds))
        if not is_idx:
            meds = [self.med_voc.word2idx[cur] for cur in meds]
        # pdb.set_trace()
        subgt, subpred, subprob = self.gt[:, meds], self.pred[:, meds], self.prob[:, meds]
        ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(
            subgt, subpred, subprob)

        return ja, prauc, avg_p, avg_r, avg_f1

    def _jaccard(self, y_gt, y_pred):
        target = np.where(y_gt == 1)[0]
        out_list = np.where(y_pred == 1)[0]
        inter = set(out_list) & set(target)
        union = set(out_list) | set(target)
        jaccard_score = 0 if len(union) == 0 else len(inter) / len(union)
        return jaccard_score

    def get_metric_for_one_med(self, gt, pred, prob):
        ja = self._jaccard(gt, pred)
        f1 = f1_score(gt, pred)
        prauc = average_precision_score(gt, prob)
        return ja, f1, prauc

    def get_metric_by_freqs(self, idx="freq_desc", words=None):
        metrics = []
        for i in range(self.gt.shape[1]):
            cur_gt, cur_pred, cur_prob = self.gt[:, i], self.pred[:, i], self.prob[:, i]
            cur_res = self.get_metric_for_one_med(cur_gt, cur_pred, cur_prob)
            metrics.append(np.array(list(cur_res)))
        metrics = np.array(metrics)
        if idx == "freq_desc":
            freqordered_metrics = metrics[list(self.medidx_ordered_desc)]
            return freqordered_metrics
        elif idx == "norm":
            return  metrics
        elif idx == 'words':
            idx = [self.med_voc.word2idx[w] for w in words]
            return metrics[idx]
        return metrics

    def check_wrong(self, tar, cur):
        res = []
        for i in range(self.gt.shape[0]):
            if self.gt[i][tar] == 0 and self.gt[i][cur] == 0:
                continue
            elif self.gt[i][tar] * self.gt[i][cur] == 1:
                continue
            if self.gt[i][tar] == 1 and self.pred[i][tar] == 0 and self.pred[i][cur] == 1:
                # pdb.set_trace()
                gt_list = np.nonzero(self.gt[i])[0]
                pred_list = np.nonzero(self.pred[i])[0]
                res.append(i)
            if self.gt[i][cur] == 1 and self.pred[i][cur] == 0 and self.pred[i][tar] == 1:
                # pdb.set_trace()
                gt_list = np.nonzero(self.gt[i])[0]
                pred_list = np.nonzero(self.pred[i])[0]
                res.append(i)
        return res

    def run(self, ops="", **kwargs):
        ja, prauc, avg_p, avg_r, avg_f1 = self.get_metric_res()
        ddi_rate = -1  # if 'd' not in ops else 0.666
        if 'd' in ops:
            pred = self.pred
            gt = self.gt
            list_pred = []
            list_target = []
            for i in range(pred.shape[0]):
                idx = np.nonzero(pred[i])[0].tolist()
                list_pred.append(idx)
            for i in range(gt.shape[0]):
                idx = np.nonzero(gt[i])[0].tolist()
                list_target.append(idx)
            # ddi_rate = ddi_rate_score([list_pred], [list_target], self.ddi_adj, self.ehr_adj)
            ddi_rate = dangerous_pair_num([list_pred], [list_target], self.ddi_adj, self.ehr_adj) # 其实是风险数字

        visit_cnt = self.gt.shape[0]
        med_cnt = self.pred.sum()
        print('adverse DDI number: {:.4f}, Jaccard: {:.4f},  PRAUC: {:.4f}, AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}\n'.format(
            ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
        ))
        if 'c' in ops:
            # idx = ['0054-3194', '0517-1010', '66213-423', '66213-425', '0225-0295', '66213-423', '66213-425']
            idx = ['0456-4020', '54838-540', '60505-2519', '0456-2010', '51079-543']
            c_ja, c_prauc, c_avg_p, c_avg_r, c_avg_f1 = self.get_metric_res_for_cert_meds(idx, False)
            print("---Certain meds metrics: ---")
            print('DDI Rate: {:.4f}, Jaccard: {:.4f},  PRAUC: {:.4f}, AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}\n'.format(
                -1, np.mean(c_ja), np.mean(c_prauc), np.mean(c_avg_p), np.mean(c_avg_r), np.mean(c_avg_f1), -1
            ))
        if 'l' in ops:
            idx = self.lowfreqmedidx  # low frequency med indices or words
            c_ja, c_prauc, c_avg_p, c_avg_r, c_avg_f1 = self.get_metric_res_for_cert_meds(idx)
            print("---Certain meds metrics: ---")
            print('DDI Rate: {:.4f}, Jaccard: {:.4f},  PRAUC: {:.4f}, AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}\n'.format(
                -1, np.mean(c_ja), np.mean(c_prauc), np.mean(c_avg_p), np.mean(c_avg_r), np.mean(c_avg_f1), -1
            ))
        if 'D' in ops:
            # topk precision
            topk = kwargs['topk']
            prob, gt = torch.Tensor(self.prob), torch.Tensor(self.gt)
            topk_val, topk_idx = torch.topk(prob, topk, dim=1)
            totol_label_num, hit_label_num = 0, 0
            # pdb.set_trace()
            for i in range(gt.shape[0]):
                cur_label = torch.nonzero(gt[i])[:, 0].tolist()
                totol_label_num += len(cur_label)
                cur_pred = topk_idx[i].tolist()
                cur_hit = [cur for cur in cur_pred if cur in set(cur_label)]
                hit_label_num += len(cur_hit)  
            acc = hit_label_num / totol_label_num
            print("topk-hit-acc: topk={}, acc={:.4f}".format(topk, acc))

        return ja, prauc, avg_p, avg_r, avg_f1


class ICD9L3:
    def __init__(self, diag_voc) -> None:
        self.word2idx = {}
        self.idx2word = {}
        self.icd2icdl3 = {}
        self.diag_voc = diag_voc
        for w in diag_voc.word2idx:
            icd_l3 = self.convert_to_3digit_icd9(w)
            self.icd2icdl3[w] = icd_l3
            if icd_l3 not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[icd_l3] = idx
                self.idx2word[idx] = icd_l3

    def convert_to_3digit_icd9(self, dxStr):
        if dxStr.startswith('E'):
            if len(dxStr) > 4: return dxStr[:4]
            else: return dxStr
        else:
            if len(dxStr) > 3: return dxStr[:3]
            else: return dxStr
    
    def get_lv3_list(self, icd_list):
        res = [self.word2idx[self.icd2icdl3[self.diag_voc.idx2word[icd_idx]]] for icd_idx in icd_list]
        res = list(set(res))
        return res

def create_atoms(mol, atom_dict):
    """Transform the atom types in a molecule (e.g., H, C, and O)
    into the indices (e.g., H=0, C=1, and O=2).
    Note that each atom index considers the aromaticity.
    """
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    res = [atom_dict[a] for a in atoms]
    return np.array(res)

def create_ijbonddict(mol, bond_dict):
    """Create a dictionary, in which each key is a node ID
    and each value is the tuples of its neighboring node
    and chemical bond (e.g., single and double) IDs.
    """
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]  # NOTE: check bond value for different molecular
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def extract_fingerprints(radius, atoms, i_jbond_dict,
                         fingerprint_dict, edge_dict):
    """Extract the fingerprints from a molecular graph
    based on Weisfeiler-Lehman algorithm.
    """

    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges.
            The updated node IDs are the fingerprint IDs.
            """
            nodes_ = list(tuple(nodes))
            # molecular graph maybe unconnected graph
            for i in range(len(nodes)):
                j_edge = i_jedge_dict[i]  # type(i_jedge_dict): defaultdict(list)
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_[i] = fingerprint_dict[fingerprint]

            """Also update each edge ID considering
            its two nodes on both sides.
            """
            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))

            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

    return np.array(nodes)


def extract_ecfp_fingerprints(m1, radius, fingerprint_dict):
    info = {}
    fp_explain = AllChem.GetMorganFingerprint(m1, radius, bitInfo=info)
    fingerprints = info.keys()
    fp_codes = [fingerprint_dict[cur_fp] for cur_fp in fingerprints]
    return np.array(sorted(fp_codes))

def extract_atom_fingerprints(m1, radius, fingerprint_dict):
    res = create_atoms(m1, fingerprint_dict)
    return res

def buildMPNN_multihot(molecule, med_voc, radius=1, device="cpu:0"):

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    MPNNSet, average_index = [], []

    print (len(med_voc.items()))
    ind_list = []
    for index, ndc in sorted(med_voc.items()):
        ind_list.append(index)
        smilesList = sorted(molecule[ndc])

        """Create each data with the above defined functions."""
        counter = 0 # counter how many drugs are under that ATC-3
        for smiles in smilesList:
            try:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                atoms = create_atoms(mol, atom_dict)
                molecular_size = len(atoms)
                i_jbond_dict = create_ijbonddict(mol, bond_dict)
                # fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                #                                     fingerprint_dict, edge_dict)
                fingerprints = extract_ecfp_fingerprints(
                    mol, radius, fingerprint_dict)
                adjacency = Chem.GetAdjacencyMatrix(mol)
                # if fingerprints.shape[0] == adjacency.shape[0]:
                # for _ in range(adjacency.shape[0] - fingerprints.shape[0]):
                #     fingerprints = np.append(fingerprints, 1)  # NOTE: what's that?
                # fingerprints = torch.LongTensor(fingerprints).to(device)
                adjacency = torch.FloatTensor(adjacency).to(device)
                MPNNSet.append([fingerprints.tolist(), adjacency, molecular_size])
                counter += 1
            except Exception as e:
                print(e)
                traceback.print_exc()
                # pdb.set_trace()
                continue
        average_index.append(counter)

        """Transform the above each data of numpy
        to pytorch tensor on a device (i.e., CPU or GPU).
        """
    print(ind_list)

    N_fingerprint = len(fingerprint_dict)

    # transform into projection matrix
    n_col = sum(average_index)
    n_row = len(average_index)

    average_projection = np.zeros((n_row, n_col))
    col_counter = 0
    for i, item in enumerate(average_index):
        average_projection[i, col_counter : col_counter + item] = 1 / item
        col_counter += item

    # padding fingerprint tensor
    max_fingers = max([len(MPNNSet[i][0]) for i in range(len(MPNNSet))])
    for i in range(len(MPNNSet)):
        MPNNSet[i][0] = MPNNSet[i][0] + [N_fingerprint] * (max_fingers - len(MPNNSet[i][0]))
        MPNNSet[i][0] = torch.LongTensor(MPNNSet[i][0]).to(device)
    return MPNNSet, N_fingerprint, torch.FloatTensor(average_projection)


def buildMPNN_ecfp(molecule, med_voc, radius=1, device="cpu:0"):
    """
    Actually, atoms index will be viewed as the fingerprint
    """
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    MPNNSet, average_index = [], []

    print (len(med_voc.items()))
    for index, ndc in sorted(med_voc.items()):

        smilesList = sorted(list(molecule[ndc]))

        """Create each data with the above defined functions."""
        counter = 0 # counter how many drugs are under that ATC-3
        for smiles in smilesList:
            try:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                atoms = create_atoms(mol, atom_dict)
                molecular_size = len(atoms)
                i_jbond_dict = create_ijbonddict(mol, bond_dict)
                # fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                #                                     fingerprint_dict, edge_dict)
                fingerprints = extract_atom_fingerprints(
                    mol, radius, fingerprint_dict)
                adjacency = Chem.GetAdjacencyMatrix(mol)
                # if fingerprints.shape[0] == adjacency.shape[0]:
                # for _ in range(adjacency.shape[0] - fingerprints.shape[0]):
                #     fingerprints = np.append(fingerprints, 1)  # NOTE: what's that?
                fingerprints = torch.LongTensor(fingerprints).to(device)
                adjacency = torch.FloatTensor(adjacency).to(device)
                MPNNSet.append((fingerprints, adjacency, molecular_size))
                counter += 1
            except Exception as e:
                print(e)
                traceback.print_exc()
                # pdb.set_trace()
                continue
        average_index.append(counter)

        """Transform the above each data of numpy
        to pytorch tensor on a device (i.e., CPU or GPU).
        """

    N_fingerprint = len(fingerprint_dict)

    # transform into projection matrix
    n_col = sum(average_index)
    n_row = len(average_index)

    average_projection = np.zeros((n_row, n_col))
    col_counter = 0
    for i, item in enumerate(average_index):
        average_projection[i, col_counter : col_counter + item] = 1 / item
        col_counter += item

    return MPNNSet, N_fingerprint, torch.FloatTensor(average_projection)


def buildMPNN_main(molecule, med_voc, radius=1, device="cpu:0"):
    """
    Actually, atoms index will be viewed as the fingerprint
    """
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    MPNNSet, average_index = [], []

    print (len(med_voc.items()))
    for index, ndc in sorted(med_voc.items()):

        # 一个NDC码有多个smiles
        smilesList = sorted(list(molecule[ndc]))

        """Create each data with the above defined functions."""
        counter = 0 # counter how many drugs are under that ATC-3
        cur_MPNNSet = []
        for smiles in smilesList:
            try:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                atoms = create_atoms(mol, atom_dict)
                molecular_size = len(atoms)
                i_jbond_dict = create_ijbonddict(mol, bond_dict)
                # fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                #                                     fingerprint_dict, edge_dict)
                fingerprints = extract_atom_fingerprints(
                    mol, radius, fingerprint_dict)
                adjacency = Chem.GetAdjacencyMatrix(mol)
                # if fingerprints.shape[0] == adjacency.shape[0]:
                # for _ in range(adjacency.shape[0] - fingerprints.shape[0]):
                #     fingerprints = np.append(fingerprints, 1)  # NOTE: what's that?
                fingerprints = torch.LongTensor(fingerprints).to(device)
                adjacency = torch.FloatTensor(adjacency).to(device)
                cur_MPNNSet.append((fingerprints, adjacency, molecular_size))
                counter += 1
            except Exception as e:
                print(e)
                traceback.print_exc()
                # pdb.set_trace()
                continue
        average_index.append(counter)

        cur_fp, cur_adj, cur_mol_size = list(zip(*cur_MPNNSet))
        new_fp = torch.cat(cur_fp)
        n_atom = sum(cur_mol_size)
        new_adj = torch.zeros(n_atom, n_atom)
        offset = 0
        for i in range(len(cur_mol_size)):
            cur_natom = cur_mol_size[i]
            new_adj[offset:offset+cur_natom, offset:offset+cur_natom] = cur_adj[i]
            offset += cur_natom
        MPNNSet.append((new_fp, new_adj, n_atom))


        """Transform the above each data of numpy
        to pytorch tensor on a device (i.e., CPU or GPU).
        """

    N_fingerprint = len(fingerprint_dict)

    average_index = [1/cur for cur in average_index]
    average_projection = np.diag(average_index)
    
    # padding MPNNSet
    MaxAtom = max([cur[1].shape[0] for cur in MPNNSet])
    pad_id = N_fingerprint
    for i in range(len(MPNNSet)):
        cur_fp, cur_adj, cur_mol_size = MPNNSet[i]
        new_fp = torch.full((MaxAtom,), pad_id).to(cur_fp.device)
        new_fp[:cur_fp.shape[0]] = cur_fp

        new_adj = torch.zeros(MaxAtom, MaxAtom).to(cur_adj.device)
        new_adj[:cur_mol_size, :cur_mol_size] = cur_adj

        # new_fp： 原子类型
        MPNNSet[i] = (new_fp, new_adj, MaxAtom)

    return MPNNSet, N_fingerprint, torch.FloatTensor(average_projection)


def buildMPNN(molecule, med_voc, radius=1, device="cpu:0"):

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    MPNNSet, average_index = [], []

    print (len(med_voc.items()))
    # NOTE: iteration order of med_voc.items must have the same order with ddi_mask_H.pkl
    for index, ndc in sorted(med_voc.items()):

        # NOTE: musk keep the smile in order, because iteration of set has different order! 
        smilesList = sorted(molecule[ndc])

        """Create each data with the above defined functions."""
        counter = 0 # counter how many drugs are under that ATC-3
        for smiles in smilesList:
            try:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                atoms = create_atoms(mol, atom_dict)
                molecular_size = len(atoms)
                i_jbond_dict = create_ijbonddict(mol, bond_dict)
                fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                    fingerprint_dict, edge_dict)
                adjacency = Chem.GetAdjacencyMatrix(mol)
                # if fingerprints.shape[0] == adjacency.shape[0]:
                for _ in range(adjacency.shape[0] - fingerprints.shape[0]):
                    fingerprints = np.append(fingerprints, 1)  # NOTE: what's that?
                fingerprints = torch.LongTensor(fingerprints).to(device)
                adjacency = torch.FloatTensor(adjacency).to(device)
                MPNNSet.append((fingerprints, adjacency, molecular_size))
                counter += 1
            except:
                # pdb.set_trace()
                traceback.print_exc()
        average_index.append(counter)

        """Transform the above each data of numpy
        to pytorch tensor on a device (i.e., CPU or GPU).
        """

    N_fingerprint = len(fingerprint_dict)

    # transform into projection matrix
    n_col = sum(average_index)
    n_row = len(average_index)

    average_projection = np.zeros((n_row, n_col))
    col_counter = 0
    for i, item in enumerate(average_index):
        average_projection[i, col_counter : col_counter + item] = 1 / item
        col_counter += item

    return MPNNSet, N_fingerprint, torch.FloatTensor(average_projection)


def buildMPNN_multihot_debug(molecule, med_voc, radius=1, device="cpu:0"):

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    MPNNSet, average_index = [], []

    print (len(med_voc.items()))
    ind_list = []
    for index, ndc in sorted(med_voc.items()):
        ind_list.append(index)
        smilesList = sorted(molecule[ndc])

        """Create each data with the above defined functions."""
        counter = 0 # counter how many drugs are under that ATC-3
        for smiles in smilesList:
            try:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                atoms = create_atoms(mol, atom_dict)
                molecular_size = len(atoms)
                i_jbond_dict = create_ijbonddict(mol, bond_dict)
                # fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                #                                     fingerprint_dict, edge_dict)
                fingerprints = extract_ecfp_fingerprints(
                    mol, radius, fingerprint_dict)
                adjacency = Chem.GetAdjacencyMatrix(mol)
                # if fingerprints.shape[0] == adjacency.shape[0]:
                # for _ in range(adjacency.shape[0] - fingerprints.shape[0]):
                #     fingerprints = np.append(fingerprints, 1)  # NOTE: what's that?
                # fingerprints = torch.LongTensor(fingerprints).to(device)
                adjacency = torch.FloatTensor(adjacency).to(device)
                MPNNSet.append([fingerprints.tolist(), adjacency, molecular_size, mol, smiles, ndc])
                counter += 1
            except Exception as e:
                print(e)
                traceback.print_exc()
                # pdb.set_trace()
                continue
        average_index.append(counter)

        """Transform the above each data of numpy
        to pytorch tensor on a device (i.e., CPU or GPU).
        """
    print(ind_list)

    N_fingerprint = len(fingerprint_dict)

    # transform into projection matrix
    n_col = sum(average_index)
    n_row = len(average_index)

    average_projection = np.zeros((n_row, n_col))
    col_counter = 0
    for i, item in enumerate(average_index):
        average_projection[i, col_counter : col_counter + item] = 1 / item
        col_counter += item

    # padding fingerprint tensor
    max_fingers = max([len(MPNNSet[i][0]) for i in range(len(MPNNSet))])
    for i in range(len(MPNNSet)):
        MPNNSet[i][0] = MPNNSet[i][0] + [N_fingerprint] * (max_fingers - len(MPNNSet[i][0]))
        MPNNSet[i][0] = torch.LongTensor(MPNNSet[i][0]).to(device)

    return MPNNSet, N_fingerprint, torch.FloatTensor(average_projection)

def unknow_debug():
    # ss_metric_obj = shot_sense_metric("../data_ordered")
    datadir = "../data_ordered"
    data_path = os.path.join(datadir, 'records_final.pkl')
    voc_path = os.path.join(datadir, 'voc_final.pkl')

    ehr_adj_path = os.path.join(datadir, 'ehr_adj_final.pkl')
    ddi_adj_path = os.path.join(datadir, 'ddi_A_final.pkl')
    ddi_mask_path = os.path.join(datadir, 'ddi_mask_H.pkl')
    molecule_path = os.path.join(datadir, 'idx2SMILES.pkl')
    device = torch.device('cpu')

    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    ddi_mask_H = dill.load(open(ddi_mask_path, 'rb'))
    data = dill.load(open(data_path, 'rb'))
    molecule = dill.load(open(molecule_path, 'rb')) 

    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    MPNNSet, N_fingerprint, average_projection = buildMPNN_multihot_debug(molecule, med_voc.idx2word, 2, device)
    feats = np.zeros((len(MPNNSet), N_fingerprint))
    for i in range(len(MPNNSet)):
        ind = set(MPNNSet[i][0].tolist()) - set({N_fingerprint})
        ind = list(ind)
        feats[i][ind] = 1
    feats_list = feats.tolist()
    feats_list = [tuple(sub) for sub in feats_list]
    for i in range(len(feats_list)):
        for j in range(i+1, len(feats_list)):
            if feats_list[i] == feats_list[j]:
                m_i = MPNNSet[i][-3]
                s_i = MPNNSet[i][-2]
                ndc_i = MPNNSet[i][-1]
                info1 = {}
                fp_explain = AllChem.GetMorganFingerprint(m_i, 2, bitInfo=info1)
                for k in info1:
                    info1[k] = len(info1[k])

                m_j = MPNNSet[j][-3]
                s_j = MPNNSet[j][-2]
                ndc_j = MPNNSet[i][-1]
                info2 = {}
                fp_explain = AllChem.GetMorganFingerprint(m_j, 2, bitInfo=info2)
                for k in info2:
                    info2[k] = len(info2[k])

                if info1 == info2 and s_i != s_j:
                    pic = Draw.MolsToGridImage(
                        [m_i, m_j], subImgSize=(1000, 1000), legends=[s_i, s_j])
                    pic.save("test.jpg")
                    print(s_i)
                    print(s_j)
                    print(ndc_i, ndc_j)
                    pdb.set_trace()
                    print("find")

    print("finish util.py")

if __name__ == "__main__":
    # gt, pred, prob = dill.load(open("gt_pred_prob.pkl", 'rb'))
    # gt, pred, prob = dill.load(open("gt_pred_prob_safe.pkl", 'rb'))
    # meds = [205, 107, 132, 202, 204, 37, 169, 196, 141, 206]
    # subgt, subpred, subprob = gt[:, meds], pred[:, meds], prob[:, meds]
    # res = multi_label_metric(gt, pred, prob)
    # sub_res = multi_label_metric(subgt, subpred, subprob)
    # ja, prauc, avg_p, avg_r, avg_f1 = res
    # print("ja, prauc, avg_p, avg_r, avg_f1: ", res)
    datadir = "../data_ndc_v1"
    modelname = sys.argv[1]
    # modelname = "nextrecemb_datandcv6_shuf_newrecup3_main_nowd_v2_nolnnn_re1"
    main_met: Metrics = _get_metric(datadir, modelname)
    main_met.run("d")
