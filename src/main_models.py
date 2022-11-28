from collections import defaultdict
from copy import deepcopy
import os
from tkinter.messagebox import NO
import dill
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dnc import DNC
from layers import FALayer, GCNLayer
import dgl
import math
import pdb
from torch.nn.parameter import Parameter


class Fagcn_main(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, eps, layer_num=1):
        super().__init__()
        self.g = g
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(self.g, hidden_dim, dropout))
            # self.layers.append(GCNLayer(self.g, hidden_dim, dropout))


        self.t0 = nn.Linear(in_dim, hidden_dim)
        self.t1 = nn.Linear(hidden_dim, out_dim)
        self.context_attn = nn.Linear(hidden_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t0.weight, gain=1.414)
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.context_attn.weight, gain=1.414)

    def forward(self, h, context=None):
        raw = h
        for i in range(self.layer_num):
            m = self.layers[i](h)  # eq 6

            # update h with context
            attn = torch.tanh(self.context_attn(context))  # eq 7
            m = attn * m
            
            h = self.eps * h + m  # eq 8
            h = torch.relu(h)
        return h


class MolecularGraphNeuralNetwork_record(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden, device, fingers, avg_projection, g=None, args=None):
        super().__init__()
        self.device = device
        self.avg_projection = avg_projection.to(device)
        self.embed_fingerprint = nn.Embedding(N_fingerprint+1, dim, padding_idx=N_fingerprint).to(self.device)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim).to(self.device)
                                            for _ in range(layer_hidden)])
        self.layer_hidden = layer_hidden

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = fingers
        self.fingerprints = torch.cat(fingerprints)
        self.molecular_sizes = molecular_sizes
        if g is None:
            adjacencies = [adjacencies[i].cpu() for i in range(len(adjacencies))]
            self.adjacencies = self.pad(adjacencies, 0)
            # build graph for fagcn
            edges = self.adjacencies.nonzero()
            num_nodes = self.fingerprints.shape[0]
            U, V= edges[:, 0], edges[:, 1]
            g = dgl.graph((U, V), num_nodes=num_nodes).to('cpu')
            g = dgl.to_simple(g)
            g = dgl.remove_self_loop(g)
            g = dgl.to_bidirected(g)
            dill.dump(g, open("g.pkl", 'wb'))

        g = g.to(self.device)
        deg = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        g.ndata['d'] = norm
        self.encoder: Fagcn_main = Fagcn_main(
            g, dim, dim, dim, dropout=0.5, eps=0.3, layer_num=2)

        self.beta = 1
        Nmed = avg_projection.shape[0]
        self.viewcat = nn.Linear(2*dim, dim)
        self.fc_selector = nn.Linear(dim, dim)

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch proc essing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N)))  # .to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def max(self, vectors, axis):
        max_vectors = [torch.max(v, 0).values for v in torch.split(vectors, axis)]
        return torch.stack(max_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, *rec_args):
        """
        visit_emb(:Tensor) with shape (Nbatch, dim)
        labels(:Tensor) with shape (Nbatch, Nmed) each row is a mult-hot vector
        """

        """MPNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(self.fingerprints)
        context = self.update_recemb(*rec_args)
        dim = context.shape[1]
        context = context.repeat(1, self.molecular_sizes[0])
        context = context.reshape(-1, dim)
        fingerprint_vectors = self.encoder(fingerprint_vectors, context)  # eq 7, 8 9

        # Molecular vector by sum or mean of the fingerprint vectors
        molecular_vectors = self.sum(fingerprint_vectors, self.molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)
        mpnn_emb = torch.mm(self.avg_projection, molecular_vectors)

        return mpnn_emb, 0
    
    def update_recemb(self, embeddings, med2diag, med2pro, ehradj_idx):
        diag_emb, pro_emb = embeddings[0], embeddings[1]
        Ndiag, Npro = med2diag.shape[1], med2pro.shape[1]
        diag_emb = diag_emb(torch.arange(Ndiag).to(self.device))
        pro_emb = pro_emb(torch.arange(Npro).to(self.device))
        # pdb.set_trace()
        med_diagview = torch.mm(med2diag, diag_emb)
        med_proview = torch.mm(med2pro, pro_emb)
        med_rec = torch.cat((med_diagview, med_proview), -1)
        med_rec = self.viewcat(med_rec)
        med_rec = med_rec + self.cooccu_aug(med_rec, ehradj_idx)
        return med_rec
    
    def cooccu_aug(self, context, ehr_adj):
        aug_emb = torch.mm(ehr_adj, context)
        sel_attn = self.fc_selector(context.clone()).tanh()
        aug_emb = sel_attn * aug_emb
        return aug_emb


class main_model(nn.Module):
    def __init__(self, vocab_size, ddi_adj,  encoder, ddi_encoder, 
                 emb_dim=256,
                 device=torch.device('cpu:0'),
                 use_aug=True,
                 ehr_adj=None,
                 med2diag=None,
                 med2pro=None,
                 args=None):
        super().__init__()
        self.use_aug = use_aug
        self.args = args
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.med2diag = med2diag
        self.med2pro = med2pro
        self.ehr_adj = torch.FloatTensor(ehr_adj).to(device) if ehr_adj is not None else None
        self.device = device
        self.vocab_size = vocab_size

        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i]+1, emb_dim, padding_idx=vocab_size[i]) for i in range(2)])
        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)])
        self.query = nn.Sequential(
                nn.ReLU(),
                nn.Linear(2 * emb_dim, emb_dim)
        )

        self.molecular_network = encoder

        self.MPNN_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.aug_MPNN_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.MPNN_layernorm = nn.LayerNorm(vocab_size[2])
        self.aug_MPNN_layernorm = nn.LayerNorm(vocab_size[2])
        self.fc_selector = nn.Linear(emb_dim, emb_dim)

        self.ddi_encoder = ddi_encoder

        adj_tensor = torch.tensor(ddi_adj)
        self.edge_index = adj_tensor.nonzero().t().contiguous()
        # x = np.ones((np.size(ddi_adj, 0),1))
        x = np.random.rand(np.size(ddi_adj, 0),1)
        self.x = torch.Tensor(x)
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)

        self.ddi_embedding = None

    def get_inputs(self, dataset, MaxVisit=2):
        # 将list的数据形式转换为tensor形式的
        # use the pad index to make th same length tensor
        diag_list, pro_list, med_list, med_ml_list, len_list = [], [], [], [], []
        max_visit = min(max([len(cur) for cur in dataset]), MaxVisit)
        ml_diag = max([len(dataset[i][j][0]) for i in range(len(dataset)) for j in range(len(dataset[i]))])
        ml_pro = max([len(dataset[i][j][1]) for i in range(len(dataset)) for j in range(len(dataset[i]))])
        # [v1, v2, v3] -> [v1], [v1, v2], [v1, v2, v3]
        for p in dataset:
            cur_diag = torch.full((max_visit, ml_diag), self.vocab_size[0])
            cur_pro = torch.full((max_visit, ml_pro), self.vocab_size[1])
            for ad_idx in range(len(p)):
                d_list, p_list, m_list = p[ad_idx]
                if ad_idx >= max_visit:
                    cur_diag[:-1] = cur_diag[1:]
                    cur_pro[:-1] = cur_pro[1:]
                    cur_diag[-1] = self.vocab_size[0]
                    cur_pro[-1] = self.vocab_size[1]
                    cur_diag[-1, :len(d_list)] = torch.LongTensor(d_list)
                    cur_pro[-1, :len(p_list)] = torch.LongTensor(p_list)
                    # visit len mask
                    len_list.append(max_visit)
                else:
                    cur_diag[ad_idx, :len(d_list)] = torch.LongTensor(d_list) 
                    cur_pro[ad_idx, :len(p_list)] = torch.LongTensor(p_list)
                    # visit len mask
                    len_list.append(ad_idx + 1)

                diag_list.append(cur_diag.long().clone())
                pro_list.append(cur_pro.long().clone())
                # bce target
                cur_med = torch.zeros(self.vocab_size[2])
                cur_med[m_list] = 1
                med_list.append(cur_med)
                # multi-label margin target
                cur_med_ml = torch.full((self.vocab_size[2],), -1)
                cur_med_ml[:len(m_list)] = torch.LongTensor(m_list)
                med_ml_list.append(cur_med_ml)


        diag_tensor = torch.stack(diag_list).to(self.device)
        pro_tensor = torch.stack(pro_list).to(self.device)
        med_tensor_bce_target = torch.stack(med_list).to(self.device)
        med_tensor_ml_target = torch.stack(med_ml_list).to(self.device)
        len_tensor = torch.LongTensor(len_list).to(self.device)

        return diag_tensor, pro_tensor, med_tensor_bce_target, med_tensor_ml_target, len_tensor
    
    def get_batch(self, data, batchsize=None):
        # diag_tensor, pro_tensor, med_tensor, len_tensor
        # data = self.get_inputs(dataset)
        if batchsize is None:
            yield data
        else:
            N = data[0].shape[0]
            idx = np.arange(N)
            np.random.shuffle(idx)
            i = 0
            while i < N:
                cur_idx = idx[i:i+batchsize]
                res = [cur_data[cur_idx] for cur_data in data]
                yield res
                i += batchsize

    def _get_query(self, diag, pro, visit_len):
        diag_emb_seq = self.dropout(self.embeddings[0](diag).sum(-2))
        pro_emb_seq = self.dropout(self.embeddings[1](pro).sum(-2))
        o1, h1 = self.encoders[0](diag_emb_seq)
        o2, h2 = self.encoders[1](pro_emb_seq)  # o2 with shape (B, M, D)
        # NOTE: select by len
        # o1, o2 with shape (B, D)
        o1 = torch.stack([o1[i,visit_len[i]-1, :] for i in range(visit_len.shape[0])])
        o2 = torch.stack([o2[i,visit_len[i]-1, :] for i in range(visit_len.shape[0])])

        patient_representations = torch.cat([o1, o2], dim=-1)  # (B, dim*2)
        query = self.query(patient_representations)  # (B, dim)

        norm_of_query = torch.norm(query, 2, 1, keepdim=True)
        normed_query = (norm_of_query / (1 + norm_of_query)) * (query / norm_of_query)       
        return query, normed_query

    def forward(self, input, visit_len):
        """
        Args:
            input(:list) with shape [(B, M, N_x)]. x can be diag, pro, med 
            len(:list/LongTensor) with shape (B, 1)
        """
        diag, pro, labels = input
        query, normed_query = self._get_query(diag, pro, visit_len)  # (Batch, dim)

        MPNN_emb, rec_loss = self.molecular_network(self.embeddings, self.med2diag, self.med2pro, self.ehr_adj)  # (N_medication, dim)

        if self.ddi_encoder:
            ddi_embedding = self.ddi_encoder(self.x, self.edge_index)
            self.ddi_embedding = ddi_embedding
            # print("self.ddi_embedding", self.ddi_embedding)
            MPNN_emb += ddi_embedding

        #  cosine samilarity
        # MPNN_emb: (M, dim), normed_query (dim,)
        normed_MPNN_emb = MPNN_emb / torch.norm(MPNN_emb, 2, 1, keepdim=True)
        # normed_MPNN_emb = self.ddi_embedding
        # print("normed_MPNN_emb", normed_MPNN_emb)
        MPNN_match = (torch.mm(normed_query, normed_MPNN_emb.t()))  # (B, N_med)
        MPNN_att = self.MPNN_layernorm(MPNN_match)
        
        result = MPNN_att  # result: (M,)

        if self.args.ddi: 
            neg_pred_prob = F.sigmoid(result)
            tmp_left = neg_pred_prob.unsqueeze(2)  # (B, Nmed, 1)
            tmp_right = neg_pred_prob.unsqueeze(1)  # (B, 1, Nmed)
            neg_pred_prob = torch.matmul(tmp_left, tmp_right)  # (N, Nmed, Nmed)
            batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        else:
            batch_neg = 0

        return result, batch_neg, 0

    def save_embedding(self):
        MPNN_emb, rec_loss = self.molecular_network(self.embeddings, self.med2diag, self.med2pro, self.ehr_adj)
        normed_MPNN_emb = MPNN_emb / torch.norm(MPNN_emb, 2, 1, keepdim=True)
        med_emb = MPNN_emb.detach().cpu().numpy()
        normed_med_emb = normed_MPNN_emb.detach().cpu().numpy()


        diag_emb = self.embeddings[0].weight[:-1]  # .detach().cpu().numpy()
        print("save no pad diag_emb: {} -> {}".format(self.embeddings[0].weight.shape, diag_emb.shape))
        normed_diag_emb = diag_emb / torch.norm(diag_emb, 2, 1, keepdim=True)
        diag_emb = diag_emb.detach().cpu().numpy()
        normed_diag_emb = normed_diag_emb.detach().cpu().numpy()

        pro_emb = self.embeddings[1].weight[:-1].detach().cpu().numpy()

        diag_file = os.path.join('saved', self.args.model_name, 'diag.tsv')
        normed_diag_file = os.path.join('saved', self.args.model_name, 'diag_normed.tsv')
        pro_file = os.path.join('saved', self.args.model_name, 'pro.tsv')
        med_file = os.path.join('saved', self.args.model_name, 'med.tsv')
        normed_med_file = os.path.join('saved', self.args.model_name, 'med_normed.tsv')

        if self.ddi_embedding != None:
            normed_ddi_embedding = self.ddi_embedding / torch.norm(self.ddi_embedding, 2, 1, keepdim=True)
            normed_ddi_emb = normed_ddi_embedding.detach().cpu().numpy()
            ddi_emb = self.ddi_embedding.detach().cpu().numpy()
            normed_ddi_file = os.path.join('saved', self.args.model_name, 'ddi_normed.tsv')
            ddi_file = os.path.join('saved', self.args.model_name, 'ddi_emb.tsv')
            np.savetxt(normed_ddi_file, normed_ddi_emb, fmt="%.4f", delimiter='\t')
            np.savetxt(ddi_file, ddi_emb, fmt="%.4f", delimiter='\t')

        np.savetxt(diag_file, diag_emb, fmt="%.4f", delimiter='\t')
        np.savetxt(normed_diag_file, normed_diag_emb, fmt="%.4f", delimiter='\t')

        np.savetxt(pro_file, pro_emb, fmt="%.4f", delimiter='\t')

        np.savetxt(med_file, med_emb, fmt="%.4f", delimiter='\t')
        np.savetxt(normed_med_file, normed_med_emb, fmt="%.4f", delimiter='\t')

        print("saved embedding files")
        return

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
            item.weight.data[:, -1] = 0.
