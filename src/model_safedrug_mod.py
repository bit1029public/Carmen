from numpy.lib.function_base import average
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import os
from dnc import DNC
from layers import GraphConvolution
import math
from torch.nn.parameter import Parameter
from models import MaskLinear


class MolecularGraphNeuralNetwork_mod(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden, device):
        super().__init__()
        self.device = device
        self.embed_fingerprint = nn.Embedding(N_fingerprint, dim).to(self.device)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim).to(self.device)
                                            for _ in range(layer_hidden)])
        self.layer_hidden = layer_hidden

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch proc essing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.mm(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, inputs):

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """MPNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            # fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.
            fingerprint_vectors = hs

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors


class SafeDrugModel_mod(nn.Module):
    def __init__(self, vocab_size, ddi_adj, ddi_mask_H, MPNNSet, N_fingerprints, average_projection, emb_dim=256, device=torch.device('cpu:0'), args=None):
        super().__init__()
        self.args = args
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

        # bipartite local embedding
        self.bipartite_transform = nn.Sequential(
            nn.Linear(emb_dim, ddi_mask_H.shape[1])
        )
        self.bipartite_output = MaskLinear(ddi_mask_H.shape[1], vocab_size[2], False)
        
        # MPNN global embedding
        self.MPNN_molecule_Set = list(zip(*MPNNSet))

        self.mol_gnn = MolecularGraphNeuralNetwork_mod(N_fingerprints, emb_dim, layer_hidden=2, device=device)
        self.average_projection = average_projection.to(device)

        # self.MPNN_emb = torch.tensor(self.MPNN_emb, requires_grad=True)
        self.MPNN_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.MPNN_layernorm = nn.LayerNorm(vocab_size[2])
        
        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)
        self.init_weights()

    def get_inputs(self, dataset):
        # use the pad index to make th same length tensor
        diag_list, pro_list, med_list, med_ml_list, len_list = [], [], [], [], []
        max_visit = max([len(cur) for cur in dataset])
        ml_diag = max([len(dataset[i][j][0]) for i in range(len(dataset)) for j in range(len(dataset[i]))])
        ml_pro = max([len(dataset[i][j][1]) for i in range(len(dataset)) for j in range(len(dataset[i]))])
        # ml_med = max([len(dataset[i][j][2]) for i in range(len(dataset)) for j in range(len(dataset[i]))])
        for p in dataset:
            cur_diag = torch.full((max_visit, ml_diag), self.vocab_size[0])
            cur_pro = torch.full((max_visit, ml_pro), self.vocab_size[1])
            for ad_idx in range(len(p)):
                d_list, p_list, m_list = p[ad_idx]
                cur_diag[ad_idx, :len(d_list)] = torch.LongTensor(d_list) 
                cur_pro[ad_idx, :len(p_list)] = torch.LongTensor(p_list)
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
                # visit len mask
                len_list.append(ad_idx + 1)

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

    def forward(self, input, visit_len):

	    # patient health representation
        diag, pro, labels = input
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

        self.MPNN_emb = self.mol_gnn(self.MPNN_molecule_Set)
        self.MPNN_emb = torch.mm(self.average_projection, self.MPNN_emb)

	    # MPNN embedding
        MPNN_match = F.sigmoid(torch.mm(query, self.MPNN_emb.t()))
        MPNN_att = self.MPNN_layernorm(MPNN_match + self.MPNN_output(MPNN_match))
        
	    # local embedding
        bipartite_emb = self.bipartite_output(F.sigmoid(self.bipartite_transform(query)), self.tensor_ddi_mask_H.t())
        
        result = torch.mul(bipartite_emb, MPNN_att)
        
        neg_pred_prob = F.sigmoid(result)
        # pdb.set_trace()
        tmp_left = neg_pred_prob.unsqueeze(2)  # (B, N, 1)
        tmp_right = neg_pred_prob.unsqueeze(1)  # (B, 1, N)
        neg_pred_prob = torch.matmul(tmp_left, tmp_right)
        # neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg

    def save_embedding(self):
        self.MPNN_emb = self.mol_gnn(self.MPNN_molecule_Set)
        MPNN_emb = torch.mm(self.average_projection, self.MPNN_emb)

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