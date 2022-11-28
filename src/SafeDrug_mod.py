import dill
import numpy as np
import pdb
import argparse
from collections import defaultdict
from sklearn.metrics import jaccard_score
from torch.optim import Adam
import os
import torch
import time
# from models import SafeDrugModel
from model_safedrug_mod import SafeDrugModel_mod as SafeDrugModel
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, buildMPNN
from util import Metrics
import torch.nn.functional as F

# torch.manual_seed(1203)
# np.random.seed(2048)
# torch.set_num_threads(30)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# setting
model_name = 'SafeDrug'
# resume_path = 'saved/{}/Epoch_49_TARGET_0.06_JA_0.5183_DDI_0.05854.model'.format(model_name)
resume_path = 'Epoch_49_TARGET_0.06_JA_0.5183_DDI_0.05854.model'

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--noddi', action='store_true', default=False, help="noddi")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_path, help='resume path')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
# parser.add_argument('--target_ddi', type=float, default=0.06, help='target ddi')
parser.add_argument('--target_ddi', type=float, default=0.005, help='target ddi')
parser.add_argument('--kp', type=float, default=0.05, help='coefficient of P signal')
parser.add_argument('--dim', type=int, default=64, help='dimension')
parser.add_argument('--datadir', type=str, default="../data/", help='dimension')
parser.add_argument('--cuda', type=int, default=-1, help='use cuda')
parser.add_argument('--early_stop', type=int, default=30, help='early stop number')
parser.add_argument('--seed', type=int, default=1029, help='random seed')

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda > -1:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(os.path.join("saved", args.model_name)):
    os.makedirs(os.path.join("saved", args.model_name))

def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return [move_to_device(cur, device) for cur in data]


def eval(model: SafeDrugModel, data_eval, voc_size, epoch, metric_obj):
    model.eval()

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    for data_tensors in model.get_batch(data_eval, 128):
        data_tensors = move_to_device(data_tensors, model.device)
        cur_diag, cur_pro, cur_med_target, _, cur_len = data_tensors
        result, _, = model((cur_diag, cur_pro, None), cur_len)
        result = F.sigmoid(result).detach().cpu().numpy()
        preds = np.zeros_like(result)
        preds[result>=0.5] = 1
        preds[result<0.5] = 0
        visit_cnt += cur_med_target.shape[0]
        med_cnt += preds.sum()
        cur_med_target = cur_med_target.detach().cpu().numpy()
        metric_obj.feed_data(cur_med_target, preds, result)

    if args.Test:
        pass
        # model.save_embedding()
    # ddi rate
    ddi_rate = -1  # ddi_rate_score(smm_record, path=os.path.join(args.datadir, 'ddi_A_final.pkl'))
    # metric_obj.set_data(cur_med_target, preds, result, save=args.Test)
    metric_obj.set_data(save=args.Test)
    ops = 'd' if args.Test else ''
    ja, prauc, avg_p, avg_r, avg_f1 = metric_obj.run(ops=ops)
    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt


# # evaluate
# def eval(model: SafeDrugModel, data_eval, voc_size, epoch, metric_obj):
#     model.eval()
#     ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
#     med_cnt, visit_cnt = 0, 0

#     data_tensors = next(model.get_batch(data_eval))
#     cur_diag, cur_pro, cur_med_target, _, cur_len = data_tensors
#     result, _, = model((cur_diag, cur_pro, None), cur_len)
#     result = F.sigmoid(result).detach().cpu().numpy()
#     preds = np.zeros_like(result)
#     preds[result>=0.5] = 1
#     preds[result<0.5] = 0
#     cur_med_target = cur_med_target.detach().cpu().numpy()
#     visit_cnt = preds.shape[0]
#     med_cnt = preds.sum()

#     if args.Test:
#         model.save_embedding()
#     # ddi rate
#     ddi_rate = -1  # ddi_rate_score(smm_record, path=os.path.join(args.datadir, 'ddi_A_final_4.pkl'))
#     metric_obj.set_data(cur_med_target, preds, result, save=args.Test)
#     ja, prauc, avg_p, avg_r, avg_f1 = metric_obj.run()
#     return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt

def main():
    
    # load data
    data_path = os.path.join(args.datadir, 'records_final_4.pkl')
    voc_path = os.path.join(args.datadir, 'voc_final_4.pkl')
    # data_path = os.path.join(args.datadir, 'records_final.pkl')
    # voc_path = os.path.join(args.datadir, 'voc_final.pkl')

    ehr_adj_path = os.path.join(args.datadir, 'ehr_adj_final_4.pkl')
    ddi_adj_path = os.path.join(args.datadir, 'ddi_A_final_4.pkl')
    ddi_mask_path = os.path.join(args.datadir, 'ddi_mask_H.pkl')
    molecule_path = os.path.join(args.datadir, 'ndc2SMILES.pkl')
    device = torch.device('cuda:'+str(args.cuda) if args.cuda > -1 else 'cpu')

    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    ddi_mask_H = dill.load(open(ddi_mask_path, 'rb'))
    data = dill.load(open(data_path, 'rb'))
    molecule = dill.load(open(molecule_path, 'rb')) 

    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    metric_obj = Metrics(data, med_voc, args)

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    MPNNSet, N_fingerprint, average_projection = buildMPNN(molecule, med_voc.idx2word, 2, device)
    print(f"N_fingerprint: {N_fingerprint}")
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = SafeDrugModel(voc_size, ddi_adj, ddi_mask_H, MPNNSet, N_fingerprint, average_projection, emb_dim=args.dim, device=device, args=args)
    # model.load_state_dict(torch.load(open(args.resume_path, 'rb')))


    if args.Test:
        model.load_state_dict(torch.load(open(args.resume_path, 'rb')))
        model.to(device=device)
        tic = time.time()
        data_test_tensors = model.get_inputs(data_test)

        ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []
        # ###
        # for threshold in np.linspace(0.00, 0.20, 30):
        #     print ('threshold = {}'.format(threshold))
        #     ddi, ja, prauc, _, _, f1, avg_med = eval(model, data_test, voc_size, 0, threshold)
        #     ddi_list.append(ddi)
        #     ja_list.append(ja)
        #     prauc_list.append(prauc)
        #     f1_list.append(f1)
        #     med_list.append(avg_med)
        # total = [ddi_list, ja_list, prauc_list, f1_list, med_list]
        # with open('ablation_ddi.pkl', 'wb') as infile:
        #     dill.dump(total, infile)
        # ###
        result = []
        for _ in range(1):
            # test_sample = np.random.choice(data_test, round(len(data_test) * 0.8), replace=True)
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_test_tensors, voc_size, 0, metric_obj)
            result.append([ddi_rate, ja, avg_f1, prauc, avg_med])
        model.save_embedding()

        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

        print (outstring)

        print ('test time: {}'.format(time.time() - tic))
        return 

    model.to(device=device)
    # print('parameters', get_n_params(model))
    # exit()
    optimizer = Adam(list(model.parameters()), lr=args.lr)

    # start iterations
    history = defaultdict(list)
    best_epoch, best_ja = 0, 0
    data_train_tensors = model.get_inputs(data_train)
    data_eval_tensors = model.get_inputs(data_eval)

    EPOCH = 400
    for epoch in range(EPOCH):
        tic = time.time()
        print ('\nepoch {} --------------------------'.format(epoch + 1))
        
        model.train()
        step = 0
        trian_visit_num = sum([len(p) for p in data_train])
        for cur_batch in model.get_batch(data_train_tensors, 16):
            cur_diag, cur_pro, cur_med_bce_target, cur_med_ml_target, cur_len = cur_batch
            result, loss_ddi = model((cur_diag, cur_pro, cur_med_bce_target), cur_len)
            # NOTE: batch of these loss function
            loss_bce = F.binary_cross_entropy_with_logits(result, cur_med_bce_target)
            loss_multi = F.multilabel_margin_loss(F.sigmoid(result), cur_med_ml_target)

            result = F.sigmoid(result).detach().cpu().numpy()
            result[result >= 0.5] = 1
            result[result < 0.5] = 0
            # get list label
            y_ll = []
            for i in range(result.shape[0]):
                curr = result[i]
                y_label = np.where(curr == 1)[0].tolist()
                y_ll.append(y_label)

            current_ddi_rate = 0 if args.noddi else ddi_rate_score([y_ll], ddi_adj)
            if current_ddi_rate <= args.target_ddi or args.noddi:
                loss = 0.95 * loss_bce + 0.05 * loss_multi # + loss_ddi # github update
            else:
                beta = min(0, 1 + (args.target_ddi - current_ddi_rate) / args.kp)
                loss = beta * (0.95 * loss_bce + 0.05 * loss_multi) + (1 - beta) * loss_ddi

            optimizer.zero_grad()
            loss.backward()  # retain_graph=True
            optimizer.step()

            step += cur_diag.shape[0]
            llprint('\rtraining step: {} / {}'.format(step, trian_visit_num))       

        print ()
        tic2 = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_eval_tensors, voc_size, epoch, metric_obj)
        print ('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))

        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        if epoch >= 5:
            print ('ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}'.format(
                np.mean(history['ddi_rate'][-5:]),
                np.mean(history['med'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['prauc'][-5:])
                ))

        if best_ja < ja:
            best_epoch = epoch
            best_ja = ja
            torch.save(model.state_dict(), open(os.path.join('saved', args.model_name, "best.model"), 'wb'))

        print ('best_epoch: {}'.format(best_epoch))

        if epoch - best_epoch > args.early_stop:
            print("Early Stop...")
            break

    dill.dump(history, open(os.path.join('saved', args.model_name, 'history_{}.pkl'.format(args.model_name)), 'wb'))

if __name__ == '__main__':
    main()
