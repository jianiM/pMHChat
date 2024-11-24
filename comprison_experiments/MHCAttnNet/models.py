# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 21:53:57 2024
@author: amber
"""

import os
import math
import sys
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import torchtext


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, context_dim):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.context_dim = context_dim
        self.tanh = nn.Tanh()

        weight = torch.zeros(feature_dim, context_dim)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)
        self.b = nn.Parameter(torch.zeros(step_dim, context_dim))

        u = torch.zeros(context_dim, 1)
        nn.init.kaiming_uniform_(u)
        self.context_vector = nn.Parameter(u)

    def forward(self, x):
        eij = torch.matmul(x, self.weight)
        # eij = [batch_size, seq_len, context_dim]
        eij = self.tanh(torch.add(eij, self.b))
        # eij = [batch_size, seq_len, context_dim]
        v = torch.exp(torch.matmul(eij, self.context_vector))  # dot product
        # v = [batch_size, seq_len, 1]
        v = v / (torch.sum(v, dim=1, keepdim=True))
        # v = [batch_size, seq_len, 1]
        weighted_input = x * v
        # weighted_input = [batch_size, seq_len, 2*hidden_dim]             -> 2 : bidirectional
        s = torch.sum(weighted_input, dim=1)
        # s = [batch_size, 2*hidden_dim]                                   -> 2 : bidirectional
        return s


class MHCAttnNet(nn.Module):
    def __init__(self, mhcvec, pepvec, mhc_weight, pep_weight): 
        super(MHCAttnNet, self).__init__()
        self.mhcvec = mhcvec 
        self.pepvec = pepvec 
        self.mhc_weight = mhc_weight 
        self.pep_weight = pep_weight 
        
        self.mhc_embed = torch.tensor(self.mhcvec.m.wv.vectors, dtype=torch.float32)
        self.mhc_embed_layer = nn.Embedding.from_pretrained(self.mhc_embed, freeze=False)    

        self.pep_embed = torch.tensor(self.pepvec.m.wv.vectors, dtype=torch.float32) 
        self.pep_embed_layer = nn.Embedding.from_pretrained(self.pep_embed, freeze=False)     

        self.mhc_lstm = nn.LSTM(input_size=100, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
        self.pep_lstm = nn.LSTM(input_size=100, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)

        self.mhc_attn = Attention(2*64, 34, 8)
        self.pep_attn = Attention(2*64, 5, 8) 

        self.mhc_linear = nn.Linear(2*64, 16) 
        self.pep_linear = nn.Linear(2*64, 16)

        self.out = nn.Linear(2*16,1) 

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)



    def forward(self, peptide, mhc):
        pep_emb = self.pep_embed_layer(peptide)        
        mhc_emb = self.mhc_embed_layer(mhc)
        # sen_emb = [batch_size, seq_len, emb_dim]
        # print('pep_emb:',pep_emb.size())
        # print('mhc_emb:',mhc_emb.size())

        pep_lstm_output, (pep_last_hidden_state, pep_last_cell_state) = self.pep_lstm(pep_emb)
        mhc_lstm_output, (mhc_last_hidden_state, mhc_last_cell_state) = self.mhc_lstm(mhc_emb)
        # sen_lstm_output = [batch_size, seq_len, 2*hidden_dim]            -> 2 : bidirectional
        # sen_last_hidden_state = [2*num_layers, batch_size, hidden_dim]   -> 2 : bidirectional
        # print('pep_lstm_output:',pep_lstm_output.size())   #  ([batch, 34, 128])
        # print('mhc_lstm_output:',mhc_lstm_output.size())

        pep_attn_linear_inp = self.pep_attn(pep_lstm_output)
        mhc_attn_linear_inp = self.mhc_attn(mhc_lstm_output)
        # sen_attn_linear_inp = [batch_size, 2*hidden_dim]                 -> 2 : bidirectional
        # print('pep_attn_linear_inp:',pep_attn_linear_inp.size())   
        # print('mhc_attn_linear_inp:',mhc_attn_linear_inp.size())
        
        # pep_last_hidden_state = pep_last_hidden_state.transpose(0, 1).contiguous().view(config.batch_size, -1)
        # mhc_last_hidden_state = mhc_last_hidden_state.transpose(0, 1).contiguous().view(config.batch_size, -1)
        # sen_last_hidden_state = [batch_size, 2*num_layers*hidden_dim]    -> 2 : bidirectional

        pep_linear_out = self.relu(self.pep_linear(pep_attn_linear_inp))
        pep_linear_out = self.dropout(pep_linear_out)
        
        mhc_linear_out = self.relu(self.mhc_linear(mhc_attn_linear_inp))
        mhc_linear_out = self.dropout(mhc_linear_out)

        conc = torch.cat((pep_linear_out, mhc_linear_out), dim=1)
        # conc = [batch_size, 2*LINEAR1_OUT]
        out = self.out(conc)
        return out





