import torch
import torch.nn as nn
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
from collections import defaultdict

import sys
sys.path.append("..")
from models import Retain
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params

model_name = 'Retain'
resume_path = 'Epoch_50_JA_0.4952_DDI_0.08157.model'

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"



# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--datadir', type=str, default="../data_ordered/", help='dimension')
parser.add_argument('--cuda', type=int, default=-1, help='use cuda')
parser.add_argument('--epoch', type=int, default=400, help='# of epoches')
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_path, help='resume path')
parser.add_argument('--early_stop', type=int, default=30, help='early stop number')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--target_ddi', type=float, default=0.06, help='target ddi')
parser.add_argument('--kp', type=float, default=0.05, help='coefficient of P signal')
parser.add_argument('--dim', type=int, default=64, help='dimension')
parser.add_argument('--seed', type=int, default=1029, help='use cuda')

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

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

        if len(input) < 2: continue
        for i in range(1, len(input)):
            target_output = model(input[:i])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[input[i][2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prob
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # prediction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= 0.4] = 1
            y_pred_tmp[y_pred_tmp < 0.4] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(y_pred_label_tmp)
            med_cnt += len(y_pred_label_tmp)
            visit_cnt += 1

        smm_record.append(y_pred_label)
        metric_obj.feed_data(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 =\
                multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        
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

    model = Retain(voc_size, device=device)
    # model.load_state_dict(torch.load(open(os.path.join("saved", args.model_name, args.resume_path), 'rb')))

    if args.Test:
        checkpoint = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
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

    print('parameters', get_n_params(model))
    optimizer = Adam(model.parameters(), args.lr)
    model.to(device=device)

    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = args.epoch
    for epoch in range(EPOCH):
        tic = time.time()
        print ('\nepoch {} --------------------------'.format(epoch + 1))
        
        model.train()
        for step, input in enumerate(data_train):
            if len(input) < 2: continue

            loss = 0
            for i in range(1, len(input)):
                target = np.zeros((1, voc_size[2]))
                target[:, input[i][2]] = 1

                output_logits = model(input[:i])
                loss += F.binary_cross_entropy_with_logits(output_logits, torch.FloatTensor(target).to(device))

            optimizer.zero_grad()
            loss.backward()
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

        if  best_ja < ja:
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


if __name__ == '__main__':
    main()
