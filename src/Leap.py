import torch
import torch.nn as nn
import argparse
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
import dill
import time
import argparse
from util import Metrics
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
import random
from collections import defaultdict

import sys
sys.path.append("..")
from models import Leap
from util import llprint, sequence_metric, sequence_output_process, ddi_rate_score, get_n_params


# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

model_name = 'Leap'
resume_path = 'saved/{}/Epoch_49_JA_0.4603_DDI_0.07427.model'.format(model_name)


parser = argparse.ArgumentParser()
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--FT', action='store_true', default=False, help="Fine Tune")
parser.add_argument('--datadir', type=str, default="../data_ordered/", help='dimension')
parser.add_argument('--ftfile', type=str, default="emm", help='finetune file')
parser.add_argument('--cuda', type=int, default=-1, help='use cuda')
parser.add_argument('--seed', type=int, default=1029, help='use cuda')
parser.add_argument('--epoch', type=int, default=400, help='# of epoches')
parser.add_argument('--early_stop', type=int, default=30, help='early stop number')
parser.add_argument('--resume_path', type=str, default=resume_path, help='resume path')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--model_name', type=str, default=model_name, help="model name")

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda > -1:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(os.path.join("saved", args.model_name)):
        os.makedirs(os.path.join("saved", args.model_name))

# evaluate
def eval(model, data_eval, voc_size, epoch, metric_obj):
    model.eval()

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    smm_record = []
    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []

        for adm_index, adm in enumerate(input):
            output_logits = model(adm)

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            output_logits = output_logits.detach().cpu().numpy()

            # prediction med set
            out_list, sorted_predict = sequence_output_process(output_logits, [voc_size[2], voc_size[2]+1])
            y_pred_label.append(sorted(sorted_predict))
            y_pred_prob.append(np.mean(output_logits[:, :-2], axis=0))

            # prediction label
            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
            visit_cnt += 1
            med_cnt += len(sorted_predict)

        smm_record.append(y_pred_label)
        metric_obj.feed_data(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = \
                sequence_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob), np.array(y_pred_label))
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step: {} / {}'.format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path=os.path.join(args.datadir, 'ddi_A_final_4.pkl'))
    print("DDI Rate: {:.4}".format(ddi_rate))
    metric_obj.set_data(save=args.Test)
    ja, prauc, avg_p, avg_r, avg_f1 = metric_obj.run()
    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt

def main():

    # load data
    data_path = os.path.join(args.datadir, 'records_final_4.pkl')
    voc_path = os.path.join(args.datadir, 'voc_final_4.pkl')
    # data_path = os.path.join(args.datadir, 'records_final.pkl')
    # voc_path = os.path.join(args.datadir, 'voc_final.pkl')
    device = torch.device('cuda:'+str(args.cuda) if args.cuda > -1 else 'cpu')

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    metric_obj = Metrics(data, med_voc, args)

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    END_TOKEN = voc_size[2] + 1

    model = Leap(voc_size, device=device)
    # model.load_state_dict(torch.load(open(args.resume_path, 'rb')))

    if args.Test:
        chk = torch.load(open(args.resume_path, 'rb'))
        model.load_state_dict(chk['model'])
        model.to(device=device)
        tic = time.time()
        result = []
        for _ in range(1):
            # test_sample = np.random.choice(data_test, round(len(data_test) * 0.8), replace=True)
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_test, voc_size, 0, metric_obj)
            result.append([ddi_rate, ja, avg_f1, prauc, avg_med])
        
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
    print('parameters', get_n_params(model))
    optimizer = Adam(model.parameters(), lr=args.lr)

    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = args.epoch
    for epoch in range(EPOCH):
        tic = time.time()
        print ('\nepoch {} --------------------------'.format(epoch + 1))

        model.train()
        for step, input in enumerate(data_train):
            for adm in input:

                loss_target = adm[2] + [END_TOKEN]
                output_logits = model(adm)
                loss = F.cross_entropy(output_logits, torch.LongTensor(loss_target).to(device))
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            llprint('\rtraining step: {} / {}'.format(step, len(data_train)))

        print ()
        tic2 = time.time() 
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_eval, voc_size, epoch, metric_obj)
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

        # torch.save(model.state_dict(), open(os.path.join('saved', args.model_name, \
        #     'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(epoch, ja, ddi_rate)), 'wb'))
        if best_ja < ja:
            best_epoch = epoch
            best_ja = ja
            savefile = os.path.join('saved', args.model_name, 'best.model')
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch}, open(savefile, 'wb'))

        print ('best_epoch: {}'.format(best_epoch))

        if epoch - best_epoch > args.early_stop:
            print("Early Stop...")
            break

        dill.dump(history, open(os.path.join('saved', args.model_name, 'history_{}.pkl'.format(args.model_name)), 'wb'))


def fine_tune(fine_tune_name=''):

    # load data
    data_path = os.path.join(args.datadir, 'records_final_4.pkl')
    voc_path = os.path.join(args.datadir, 'voc_final_4.pkl')
    # data_path = os.path.join(args.datadir, 'records_final.pkl')
    # voc_path = os.path.join(args.datadir, 'voc_final.pkl')
    device = torch.device('cuda:'+str(args.cuda) if args.cuda > -1 else 'cpu')

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    metric_obj = Metrics(data, med_voc, args)

    ddi_adj_path = os.path.join(args.datadir, 'ddi_A_final_4.pkl')
    ddi_A = dill.load(open(ddi_adj_path, 'rb'))

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = Leap(voc_size, device=device)
    model.load_state_dict(torch.load(open(args.ftfile, 'rb'))['model'])
    model.to(device)

    END_TOKEN = voc_size[2] + 1

    optimizer = Adam(model.parameters(), lr=args.lr)
    ddi_rate_record = []
    best_epoch, best_ja = 0, 0

    EPOCH = 100
    for epoch in range(EPOCH):
        loss_record = []
        start_time = time.time()
        tic = time.time()
        random_train_set = [random.choice(data_train) for i in range(len(data_train))]
        for step, input in enumerate(random_train_set):
            model.train()
            K_flag = False
            for adm in input:
                target = adm[2]
                output_logits = model(adm)
                out_list, sorted_predict = sequence_output_process(output_logits.detach().cpu().numpy(), [voc_size[2], voc_size[2] + 1])

                inter = set(out_list) & set(target)
                union = set(out_list) | set(target)
                jaccard = 0 if union == 0 else len(inter) / len(union)
                K = 0
                for i in out_list:
                    if K == 1:
                        K_flag = True
                        break
                    for j in out_list:
                        if ddi_A[i][j] == 1:
                            K = 1
                            break

                loss = -jaccard * K * torch.mean(F.log_softmax(output_logits, dim=-1))
                loss_record.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            llprint('\rtraining step: {} / {}'.format(step, len(random_train_set)))

        tic2 = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_eval, voc_size, epoch, metric_obj)
        print ('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))
        if best_ja < ja:
            best_epoch = epoch
            best_ja = ja
            savefile = os.path.join('saved', args.model_name, 'final.model')
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch}, open(savefile, 'wb'))

        if epoch - best_epoch > args.early_stop:
            print("Early Stop...")
            break

        if K_flag:
            print ()
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_test, voc_size, epoch, metric_obj)

    # test
    torch.save(model.state_dict(), open(
        os.path.join('saved', args.model_name, 'final.model'), 'wb'))


if __name__ == '__main__':
    if not args.FT:
        main()
    else:
        file = args.ftfile
        fine_tune(fine_tune_name=file)
