from ast import arg
from lib2to3.pytree import Node
import dill
import numpy as np
import argparse
from collections import defaultdict
from sklearn import metrics
from sklearn.metrics import jaccard_score
from torch.optim import Adam, SGD
import os
import pdb
import torch
import time
from main_models import main_model, MolecularGraphNeuralNetwork_record
from main_baseline import (
    MolecularGraphNeuralNetwork_fagcn,
    MolecularGraphNeuralNetwork_ContextIndependent)
from util import buildMPNN_main, buildMPNN_multihot, llprint, multi_label_metric, ddi_rate_score, get_n_params, buildMPNN, buildMPNN_ecfp
from util import Metrics, get_ehr_adj
import torch.nn.functional as F
import scipy.sparse as sp
from torch_sparse import SparseTensor
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from models_gnn import GNN
# from main_models import GCNConv

# torch.set_num_threads(30)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# setting
model_name = 'trial_model'
# resume_path = 'saved/{}/Epoch_49_TARGET_0.06_JA_0.5183_DDI_0.05854.model'.format(model_name)
resume_path = 'Epoch_49_TARGET_0.06_JA_0.5183_DDI_0.05854.model'

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--label_soft', action='store_true', default=False, help="soft label")
parser.add_argument('--focal_loss', action='store_true', default=False, help="focal loss")
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_path, help='resume path')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
# parser.add_argument('--target_ddi', type=float, default=0.01346, help='target ddi')
parser.add_argument('--target_ddi', type=float, default=0.005, help='target ddi')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--kp', type=float, default=0.05, help='coefficient of P signal')
parser.add_argument('--dim', type=int, default=64, help='dimension') # 改回64
parser.add_argument('--datadir', type=str, default="../data_ordered/", help='datadir')
parser.add_argument('--encoder', type=str, default="main", help='molecular encoder type')
parser.add_argument('--cuda', type=int, default=-1, help='use cuda')
parser.add_argument('--seed', type=int, default=1029, help='random seed')
parser.add_argument('--epoch', type=int, default=400, help='# of epoches')
parser.add_argument('--early_stop', type=int, default=30, help='early stop number')
parser.add_argument('--load', action='store_true', default=False, help='load resume file')
parser.add_argument('--noaug', action='store_true', default=False, help='do not use aug part')
parser.add_argument('--ddi', action='store_true', default=False, help='use ddi')

parser.add_argument('--ddi_encoding', action='store_true', default=False, help='use ddi encoding')
parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
parser.add_argument('--gnn_type', type=str, default="gat")
parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
parser.add_argument('--p_or_m', type=str, default="minus")
parser.add_argument('--MIMIC', type=int, default=3, help="mimic3 or mimic4")


args = parser.parse_args()
print(args)

if not os.path.exists(os.path.join("saved", args.model_name)):
        os.makedirs(os.path.join("saved", args.model_name))

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda > -1:
    torch.cuda.manual_seed(args.seed)

def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return [move_to_device(cur, device) for cur in data]


def eval(model: main_model, data_eval, voc_size, epoch, metric_obj: Metrics):
    model.eval()

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    for data_tensors in model.get_batch(data_eval, 128):
        data_tensors = move_to_device(data_tensors, model.device)
        cur_diag, cur_pro, cur_med_target, _, cur_len = data_tensors
        result, _, _ = model((cur_diag, cur_pro, None), cur_len)
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


def main():
    # load data
    if args.MIMIC == 4:
        data_path = os.path.join(args.datadir, 'records_final_4.pkl')
        voc_path = os.path.join(args.datadir, 'voc_final_4.pkl')
        ddi_adj_path = os.path.join(args.datadir, 'ddi_A_final_4.pkl') 
    else:
        data_path = os.path.join(args.datadir, 'records_final.pkl')
        voc_path = os.path.join(args.datadir, 'voc_final.pkl')
        ddi_adj_path = os.path.join(args.datadir, 'ddi_A_final.pkl')

    molecule_path = os.path.join(args.datadir, 'ndc2SMILES.pkl')
    device = torch.device('cuda:'+str(args.cuda) if args.cuda > -1 else 'cpu')

    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    data = dill.load(open(data_path, 'rb'))
    molecule = dill.load(open(molecule_path, 'rb')) 

    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    metric_obj = Metrics(data, med_voc, args)
    use_aug = not args.noaug

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]
    ehr_adj, med2diag, med2pro, g = None, None, None, None

    # print("ddi_embedding", ddi_embedding)

    def create_matrices():
        Ndiag, Npro, Nmed = voc_size
        med_count_in_train = np.zeros(Nmed)
        med2diag = np.zeros((Nmed, Ndiag))
        med2pro = np.zeros((Nmed, Npro))
        for p in data_train:
            for m in p:
                cur_diag, cur_pro, cur_med = m
                for cm in cur_med:
                    med2diag[cm][cur_diag] += 1
                    med2pro[cm][cur_pro] += 1
                    med_count_in_train[cm] += 1
        med_count_in_train[med_count_in_train==0] = 1

        DTH, PTH = 0, 0
        med2diag = torch.FloatTensor(med2diag)
        med2pro = torch.FloatTensor(med2pro)

        med2diag = med2diag / med_count_in_train.reshape(-1,1)
        med2diag = F.normalize(med2diag, p=1, dim=1)
        med2diag = med2diag.to(torch.float32).to(device)

        med2pro = med2pro / med_count_in_train.reshape(-1,1)
        med2pro = F.normalize(med2pro, p=1, dim=1)
        med2pro = med2pro.to(torch.float32).to(device)


        ehr_adj = get_ehr_adj(data_train, Nmed, no_weight=False)
        ehr_sim = torch.from_numpy(ehr_adj) / (torch.from_numpy(med_count_in_train).reshape(1,-1))
        ehr_sim = ehr_sim.to(torch.float32).to(device)
        ehr_sim = F.normalize(ehr_sim, p=1, dim=1)

        norm = torch.relu(med2pro.sum(1).reshape(-1, 1) - 1) + 1
        med2pro = med2pro / norm
        med2pro = med2pro.to(device)
        ehr_adj = get_ehr_adj(data_train, Nmed, no_weight=False)
        
        ehr_norm = ehr_adj.sum(1).reshape(-1, 1)
        ehr_norm[ehr_norm==0] = 1
        ehr_adj = ehr_adj / ehr_norm
        return ehr_adj, med2diag, med2pro, ehr_sim
 
    if args.encoder == "main":
        # build_fun 主要是为了将药物分子按照一定的格式处理成邻接矩阵的输入
        build_fun = buildMPNN_main
        encoder_cls = MolecularGraphNeuralNetwork_record
        ehr_adj, med2diag, med2pro, ehr_sim = create_matrices()
        # print ("matrices, ", ehr_adj, med2diag, med2pro, ehr_sim)
        g = None

    elif args.encoder == 'fagcn': # vanilla gnn
        # Carmen_{c-}
        build_fun = buildMPNN_ecfp
        encoder_cls = MolecularGraphNeuralNetwork_fagcn

    ddi_encoder = None
    if args.ddi_encoding:
        ddi_encoder = GNN(p_or_m=args.p_or_m, device=device, num_layer=args.num_layer, emb_dim=args.dim, gnn_type=args.gnn_type)

    MPNNSet, N_fingerprint, average_projection = build_fun(molecule,
                                                           med_voc.idx2word,
                                                           radius=2,
                                                           device=device)
    print(f"N_fingerprint: {N_fingerprint}")
    MPNN_molecule_Set = list(zip(*MPNNSet))
    encoder = encoder_cls(N_fingerprint, args.dim, 2,
                          device=device,
                          fingers=MPNN_molecule_Set,
                          avg_projection=average_projection,
                          g=g,
                          args=args)
    
    model = main_model(voc_size, ddi_adj, encoder,
                        ddi_encoder, # add ddi_encoder
                        # exist_ddi_list, # add exist ddi
                        emb_dim=args.dim,
                        device=device,
                        use_aug=use_aug,
                        # ehr_adj=ehr_sim,
                        ehr_adj=ehr_adj,
                        med2diag=med2diag,
                        med2pro=med2pro,
                        args=args)
    model.to(device=device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # model.ddi_encoding()
    # optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.95, weight_decay=1e-5)
    epoch_begin = 0

    if args.Test or args.load:
        checkpoint = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_begin = checkpoint['epoch'] + 1
        print(f"Load {args.resume_path} finish...")

    if args.Test:
    # if True:
        model.to(device=device)
        tic = time.time()
        data_test_tensors = model.get_inputs(data_test)

        result = []
        metrics = eval(model, data_test_tensors, voc_size, 0, metric_obj) # ddi_adj删了
        model.save_embedding()
        result.append(list(metrics))
        
        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

        print (outstring)

        print ('test time: {}'.format(time.time() - tic))
        return 

    # start iterations
    history = defaultdict(list)
    best_epoch, best_ja = 0, 0


    # ddi_embedding = ddi_encoding(ddi_adj, args.dim)
    # ddi_embedding = ddi_embedding.to(device)

    # 将list数据转化为tensor
    data_train_tensors = model.get_inputs(data_train)
    data_eval_tensors = model.get_inputs(data_eval)
    EPOCH = args.epoch
    for epoch in range(epoch_begin, EPOCH):
        tic = time.time()
        print ('\nepoch {} --------------------------'.format(epoch + 1))
        
        model.train()
        step = 0
        trian_visit_num = sum([len(p) for p in data_train])
        for cur_batch in model.get_batch(data_train_tensors, 16):
            cur_diag, cur_pro, cur_med_bce_target, cur_med_ml_target, cur_len = cur_batch
            result, loss_ddi, _ = model((cur_diag, cur_pro, cur_med_bce_target), cur_len)
            # NOTE: batch of these loss function
            if args.label_soft:
                cur_med_bce_target = torch.matmul(cur_med_bce_target, ehr_sim) + cur_med_bce_target
            loss_bce = F.binary_cross_entropy_with_logits(result, cur_med_bce_target)
            if args.focal_loss:
                alpha = 0.25
                gamma = 1
                BCE_loss = F.binary_cross_entropy_with_logits(result, cur_med_bce_target, reduce=False)
                pt = torch.exp(-BCE_loss)
                F_loss = alpha * (1-pt)**gamma * BCE_loss
                loss_bce = torch.mean(F_loss)*10
            loss_multi = F.multilabel_margin_loss(F.sigmoid(result), cur_med_ml_target)

            # NOTE: value range of loss_ddi 
            loss = 0.95 * loss_bce + 0.05 * loss_multi  #  + loss_ddi
            if args.ddi:
                labellist = []
                for i in range(cur_med_ml_target.shape[0]):
                    cur = torch.nonzero(cur_med_ml_target[i])[:, 0].tolist()
                    labellist.append(cur)
                cur_ddi_rate = ddi_rate_score([labellist], ddi_adj)
                if cur_ddi_rate > args.target_ddi:   # 如果当前ddi率大于目标ddi率，则加入ddi loss
                    beta = min(0, 1 + (args.target_ddi - cur_ddi_rate) / 0.05)
                    loss = beta * loss + (1 - beta) * loss_ddi
 
            optimizer.zero_grad()
            loss.backward()  # retain_graph=True
            optimizer.step()

            step += cur_diag.shape[0]
            llprint('\rtraining step: {} / {}'.format(step, trian_visit_num))

        print ()
        tic2 = time.time() 
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_eval_tensors, voc_size, epoch, metric_obj)
        print ('training time: {}, test time: {}'.format(tic2 - tic, time.time() - tic2))

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

        savefile = os.path.join('saved', args.model_name, 'Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate))
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch}, open(savefile, 'wb'))

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

if __name__ == '__main__':
    main()
