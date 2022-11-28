from collections import defaultdict
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from dnc import DNC
from layers import FALayer
import dgl
import math
from torch.nn.parameter import Parameter
# from trial_models import Fagcn


class Fagcn(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, eps, layer_num=1):
        super().__init__()
        self.g = g
        print("g is here")
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(self.g, hidden_dim, dropout))

        self.t0 = nn.Linear(in_dim, hidden_dim)
        self.t1 = nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t0.weight, gain=1.414)
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)

    def forward(self, h):
        # h = F.dropout(h, p=self.dropout, training=self.training)
        # h = torch.relu(self.t0(h))
        # h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            m = self.layers[i](h)
            h = self.eps * h + m
            h = torch.relu(h)
        # h = self.t1(h)
        return h  # F.log_softmax(h, 0)


class MolecularGraphNeuralNetwork_ContextIndependent(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden, device, fingers, avg_projection, g=None, args=None):
        super().__init__()
        self.device = device
        self.avg_projection = avg_projection.to(device)
        self.embed_fingerprint = nn.Embedding(N_fingerprint, dim).to(self.device)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim).to(self.device)
                                            for _ in range(layer_hidden)])
        self.layer_hidden = layer_hidden

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = fingers
        self.fingerprints = torch.cat(fingerprints)
        self.adjacencies = self.pad(adjacencies, 0)
        self.molecular_sizes = molecular_sizes
        # build graph for fagcn
        edges = self.adjacencies.nonzero()
        U, V= edges[:, 0], edges[:, 1]
        g = dgl.graph((U, V)).to('cpu')
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        g = dgl.to_bidirected(g)
        g = g.to(self.device)
        deg = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        g.ndata['d'] = norm
        self.encoder: Fagcn = Fagcn(
            g, dim, dim, dim, dropout=0.5, eps=0.3, layer_num=2)
        
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
        zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
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

    def forward(self, *rec_args):

        """Cat or pad each input data for batch processing."""

        """MPNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(self.fingerprints)
        context = self.update_recemb(*rec_args)
        fingerprint_vectors = self.encoder(fingerprint_vectors)

        molecular_vectors = self.sum(fingerprint_vectors, self.molecular_sizes)
        mpnn_emb = torch.mm(self.avg_projection, molecular_vectors)
        mpnn_emb += context

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


class MolecularGraphNeuralNetwork_fagcn(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden, device, fingers, avg_projection, g=None, args=None):
        super().__init__()
        self.device = device
        self.avg_projection = avg_projection.to(device)
        self.embed_fingerprint = nn.Embedding(N_fingerprint, dim).to(self.device)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim).to(self.device)
                                            for _ in range(layer_hidden)])
        self.layer_hidden = layer_hidden

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = fingers
        self.fingerprints = torch.cat(fingerprints)
        self.adjacencies = self.pad(adjacencies, 0)
        self.molecular_sizes = molecular_sizes
        # build graph for fagcn
        edges = self.adjacencies.nonzero()
        U, V= edges[:, 0], edges[:, 1]
        g = dgl.graph((U, V)).to('cpu')
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        g = dgl.to_bidirected(g)
        g = g.to(self.device)
        deg = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        g.ndata['d'] = norm
        self.encoder: Fagcn = Fagcn(
            g, dim, dim, dim, dropout=0.5, eps=0.3, layer_num=2)

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

    def forward(self, inputs, *args, **kwargs):

        """Cat or pad each input data for batch processing."""

        """MPNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(self.fingerprints)
        fingerprint_vectors = self.encoder(fingerprint_vectors)
        # for l in range(self.layer_hidden):
        #     hs = self.update(self.adjacencies, fingerprint_vectors, l)
        #     # fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.
        #     fingerprint_vectors = hs

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, self.molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)
        mpnn_emb = torch.mm(self.avg_projection, molecular_vectors)

        return mpnn_emb, 0
    
    def update_recemb(self, *args, **kwargs):
        ...
