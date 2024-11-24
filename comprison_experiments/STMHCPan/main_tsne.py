import math
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import time
import datetime
import random
from scipy.stats import pearsonr
random.seed(1234)


import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from functools import reduce
from tqdm import tqdm, trange
from copy import deepcopy

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.utils.data as Data
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def make_data(data, pep_max_len, hla_max_len):
    pep_inputs, hla_inputs, labels = [], [], []
    pep_lens = []
    for pep, hla, label in zip(data.Seq, data.HLA_sequence, data.label):
        # Pad the peptide and HLA sequences
        pep = pep[:pep_max_len]
        pep = pep.ljust(pep_max_len, '-')
        hla = hla.ljust(hla_max_len, '-')
        # Convert sequences to numerical representations
        pep_input = [vocab[n] for n in pep]
        hla_input = [vocab[n] for n in hla]
        pep_inputs.append(pep_input)
        hla_inputs.append(hla_input)
        labels.append(label)
        pep_lens.append(len(pep_input))  # Use the length of pep_input instead of a fixed value
    # Convert lists to tensors
    pep_inputs = torch.LongTensor(pep_inputs)
    hla_inputs = torch.LongTensor(hla_inputs)
    labels = torch.LongTensor(labels)
    pep_lens = torch.LongTensor(pep_lens)
    return pep_inputs, hla_inputs, labels, pep_lens

class MyDataSet(Data.Dataset):
    def __init__(self, pep_inputs, hla_inputs, labels, pep_lens):
        super(MyDataSet, self).__init__()
        self.pep_inputs = pep_inputs
        self.hla_inputs = hla_inputs
        self.labels = labels
        self.pep_lens = pep_lens

    def __len__(self): # 
        return self.pep_inputs.shape[0] 

    def __getitem__(self, idx):
#         return self.pep_inputs[idx], self.hla_inputs[idx], self.labels[idx],self.pep_lens[idx]
        return torch.cat((self.hla_inputs[idx],self.pep_inputs[idx]), dim=0), self.labels[idx], self.pep_lens[idx]

def seq_len_to_mask(seq_len, max_len=49): #50

    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask


def get_embeddings(init_embed, padding_idx=None):

    if isinstance(init_embed, tuple):
        res = nn.Embedding(num_embeddings=init_embed[0], embedding_dim=init_embed[1], padding_idx=padding_idx)
#         nn.init.uniform_(res.weight.data, a=-np.sqrt(3 / res.weight.data.size(1)),
#                          b=np.sqrt(3 / res.weight.data.size(1)))
    elif isinstance(init_embed, nn.Module):
        res = init_embed
    elif isinstance(init_embed, torch.Tensor):
        res = nn.Embedding.from_pretrained(init_embed, freeze=False)
    elif isinstance(init_embed, np.ndarray):
        init_embed = torch.tensor(init_embed, dtype=torch.float32)
        res = nn.Embedding.from_pretrained(init_embed, freeze=False)
    else:
        raise TypeError(
            'invalid init_embed type: {}'.format((type(init_embed))))
    return res


class StarTransformer(nn.Module):
    r"""
    Star-Transformer 的encoder部分。 输入3d的文本输入, 返回相同长度的文本编码
    paper: https://arxiv.org/abs/1902.09113
    """

    def __init__(self, hidden_size, num_layers, num_head, head_dim, dropout=0.1, max_len=None):
        r"""
        
        :param int hidden_size: 输入维度的大小。同时也是输出维度的大小。
        :param int num_layers: star-transformer的层数
        :param int num_head: head的数量。
        :param int head_dim: 每个head的维度大小。
        :param float dropout: dropout 概率. Default: 0.1
        :param int max_len: int or None, 如果为int，输入序列的最大长度，
            模型会为输入序列加上position embedding。
            若为`None`，忽略加上position embedding的步骤. Default: `None`
        """
        super(StarTransformer, self).__init__()
        self.iters = num_layers

        self.norm = nn.ModuleList([nn.LayerNorm(hidden_size, eps=1e-6) for _ in range(self.iters)])
        # self.emb_fc = nn.Conv2d(hidden_size, hidden_size, 1)
        self.emb_drop = nn.Dropout(dropout)
        self.ring_att = nn.ModuleList(
            [_MSA1(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.0)
             for _ in range(self.iters)])
        self.star_att = nn.ModuleList(
            [_MSA2(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.0)
             for _ in range(self.iters)])

        if max_len is not None:
            self.pos_emb = nn.Embedding(max_len, hidden_size)
        else:
            self.pos_emb = None

    def forward(self, data, mask):
        r"""
        :param FloatTensor data: [batch, length, hidden] 输入的序列
        :param ByteTensor mask: [batch, length] 输入序列的padding mask, 在没有内容(padding 部分) 为 0,
            否则为 1
        :return: [batch, length, hidden] 编码后的输出序列
                [batch, hidden] 全局 relay 节点, 详见论文
        """

        def norm_func(f, x):
            # B, H, L, 1
            return f(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        B, L, H = data.size()
        mask = (mask.eq(False))  # flip the mask for masked_fill_
        smask = torch.cat([torch.zeros(B, 1, ).byte().to(mask), mask], 1)

        embs = data.permute(0, 2, 1)[:, :, :, None]  # B H L 1
        if self.pos_emb:
            P = self.pos_emb(torch.arange(L, dtype=torch.long, device=embs.device) \
                             .view(1, L)).permute(0, 2, 1).contiguous()[:, :, :, None]  # 1 H L 1
            embs = embs + P
        embs = norm_func(self.emb_drop, embs)
        nodes = embs
        relay = embs.mean(2, keepdim=True)
        ex_mask = mask[:, None, :, None].expand(B, H, L, 1)
        r_embs = embs.view(B, H, 1, L)
#         nodes_attns = []
#         relays_attns = []
        for i in range(self.iters):
            ax = torch.cat([r_embs, relay.expand(B, H, 1, L)], 2)
            nodes = F.leaky_relu(self.ring_att[i](norm_func(self.norm[i], nodes), ax=ax))
            # nodes = F.leaky_relu(self.ring_att[i](nodes, ax=ax))
#             nodes_attns.append(nodes_att)
            relay = F.leaky_relu(self.star_att[i](relay, torch.cat([relay, nodes], 2), smask))
#             relays_attns.append(relay_att)
            nodes = nodes.masked_fill_(ex_mask, 0)

        nodes = nodes.view(B, H, L).permute(0, 2, 1)

        return nodes, relay.view(B, H)#, nodes_attns, relays_attns

class _MSA1(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1):
        super(_MSA1, self).__init__()
        # Multi-head Self Attention Case 1, doing self-attention for small regions
        # Due to the architecture of GPU, using hadamard production and summation are faster than dot production when unfold_size is very small
        self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)

        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def forward(self, x, ax=None):
        # x: B, H, L, 1, ax : B, H, X, L append features
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = x.shape

        q, k, v = self.WQ(x), self.WK(x), self.WV(x)  # x: (B,H,L,1)

        if ax is not None:
            aL = ax.shape[2]
            ak = self.WK(ax).view(B, nhead, head_dim, aL, L)
            av = self.WV(ax).view(B, nhead, head_dim, aL, L)
        q = q.view(B, nhead, head_dim, 1, L)
        k = F.unfold(k.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0)) \
            .view(B, nhead, head_dim, unfold_size, L)
        v = F.unfold(v.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0)) \
            .view(B, nhead, head_dim, unfold_size, L)
        if ax is not None:
            k = torch.cat([k, ak], 3)
            v = torch.cat([v, av], 3)

        alphas = self.drop(F.softmax((q * k).sum(2, keepdim=True) / np.sqrt(head_dim), 3))  # B N L 1 U
        att = (alphas * v).sum(3).view(B, nhead * head_dim, L, 1)

        ret = self.WO(att)

        return ret #,alphas

class _MSA2(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1):
        # Multi-head Self Attention Case 2, a broadcastable query for a sequence key and value
        super(_MSA2, self).__init__()
        self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)

        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3
    def forward(self, x, y, mask=None):
        # x: B, H, 1, 1, 1 y: B H L 1
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = y.shape

        q, k, v = self.WQ(x), self.WK(y), self.WV(y)

        q = q.view(B, nhead, 1, head_dim)  # B, H, 1, 1 -> B, N, 1, h
        k = k.view(B, nhead, head_dim, L)  # B, H, L, 1 -> B, N, h, L
        v = v.view(B, nhead, head_dim, L).permute(0, 1, 3, 2)  # B, H, L, 1 -> B, N, L, h
        pre_a = torch.matmul(q, k) / np.sqrt(head_dim)
        if mask is not None:
            pre_a = pre_a.masked_fill(mask[:, None, None, :], -float('inf'))
        alphas = self.drop(F.softmax(pre_a, 3))  # B, N, 1, L
        att = torch.matmul(alphas, v).view(B, -1, 1, 1)  # B, N, 1, h -> B, N*h, 1, 1
        return self.WO(att) #,alphas


class StarTransEnc(nn.Module):
    r"""
    带word embedding的Star-Transformer Encoder
    """

    def __init__(self, embed,
                 hidden_size,
                 num_layers,
                 num_head,
                 head_dim,
                 max_len,
                 emb_dropout,
                 dropout):
        r"""
        
        :param embed: 单词词典, 可以是 tuple, 包括(num_embedings, embedding_dim), 即
            embedding的大小和每个词的维度. 也可以传入 nn.Embedding 对象,此时就以传入的对象作为embedding
        :param hidden_size: 模型中特征维度.
        :param num_layers: 模型层数.
        :param num_head: 模型中multi-head的head个数.
        :param head_dim: 模型中multi-head中每个head特征维度.
        :param max_len: 模型能接受的最大输入长度.
        :param emb_dropout: 词嵌入的dropout概率.
        :param dropout: 模型除词嵌入外的dropout概率.
        """
        super(StarTransEnc, self).__init__()
        self.embedding = get_embeddings(embed,padding_idx=0)
        emb_dim = self.embedding.embedding_dim
        self.emb_fc = nn.Linear(emb_dim, hidden_size)
        # self.emb_drop = nn.Dropout(emb_dropout)
        self.encoder = StarTransformer(hidden_size=hidden_size,
                                       num_layers=num_layers,
                                       num_head=num_head,
                                       head_dim=head_dim,
                                       dropout=dropout,
                                       max_len=max_len)
        
#         conv_block_klass = ConvBlock
# #         Embedding Layer
#         self.stem = nn.Sequential(
#         #             Rearrange('b n d -> b d n'),
# #             Dynamic_conv1d(49, 49, 3,padding = 1),
#             Residual(conv_block_klass(49)),
# #             AttentionPool(49, pool_size = 2)
            
#         )
#         self.stem2 = nn.Sequential(
#         #             Rearrange('b n d -> b d n'),
#             nn.Conv1d(34, 34, 3,padding = 1),
#             Residual(conv_block_klass(34)),
#             AttentionPool(34, pool_size = 2)
#         )

    def forward(self, x, mask):
        r"""
        :param FloatTensor x: [batch, length, hidden] 输入的序列
        :param ByteTensor mask: [batch, length] 输入序列的padding mask, 在没有内容(padding 部分) 为 0,
            否则为 1
        :return: [batch, length, hidden] 编码后的输出序列
                [batch, hidden] 全局 relay 节点, 详见论文
        """
        x = self.embedding(x)
        x = self.emb_fc(x)
#         x = self.stem(x)
        #nodes, relay, nodes_attns, relays_attns = self.encoder(x3, mask3)
        nodes, relay = self.encoder(x, mask)
        return nodes, relay, #nodes_attns, relays_attns
        

class _Cls(nn.Module):
    def __init__(self, in_dim, num_cls, hid_dim, dropout=0.1):
        super(_Cls, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, num_cls)
        )

    def forward(self, x):
        h = self.fc(x)
        return h


class STSeqCls(nn.Module):
    def __init__(self, embed, num_cls=2,
                 hidden_size=300,
                 num_layers=1,
                 num_head=9,
                 head_dim=32,
                 max_len=512,
                 cls_hidden_size=600,
                 emb_dropout=0.1,
                 dropout=0.1):
        
        super(STSeqCls, self).__init__()
        self.enc = StarTransEnc(embed=embed,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                num_head=num_head,
                                head_dim=head_dim,
                                max_len=max_len,
                                emb_dropout=emb_dropout,
                                dropout=dropout)
        self.cls = _Cls(hidden_size, num_cls, cls_hidden_size, dropout=dropout)


    def forward(self, words, seq_len,deivce):

        mask = seq_len_to_mask(seq_len,max_len=49).to(deivce)  #to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#         mask2 = seq_len_to_mask(torch.tensor([34]*len(seq_len))).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        nodes, relay = self.enc(words, mask)
        y = 0.5 * (relay + nodes.max(1)[0])
#         y = torch.cat([relay, torch.sort(nodes,dim=1)[0][:,-1,:], torch.sort(nodes,dim=1)[0][:,-2,:]],1)
        
        output = self.cls(y)  # [bsz, n_cls]
        return output#, nodes_attns, relays_attns 
    
# y_pred : [0,1,1,0..] binary 
# y_prob: probility after sofrmax 
# add: aupr, acc, sensitivity? pcc 

def performances(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN
    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])
    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])
    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    print('auc={:.4f}|aupr={:.4f}'.format(auc[0, 0], aupr[0,0]))
    return TP,FP,FN,TN,fpr,tpr,auc[0, 0], aupr[0, 0],f1_score, accuracy, recall, specificity, precision



def transfer(y_prob, threshold = 0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])

f_mean = lambda l: sum(l)/len(l)


def train_step(model, train_loader, fold, epoch, epochs, criterion,optimizer,threshold,device):
    # device = torch.device("cuda" if use_cuda else "cpu")
    
    time_train_ep = 0
    model.train()
    y_true_train_list, y_prob_train_list = [], []
    loss_train_list = []
    for train_pep_inputs, train_pep_lens, train_labels in tqdm(train_loader):
        '''
        pep_inputs: [batch_size, pep_len]
        hla_inputs: [batch_size, hla_len]
        train_outputs: [batch_size, 2]
        '''
        train_pep_inputs, train_labels = train_pep_inputs.to(device), train_labels.to(device)
        train_pep_lens = train_pep_lens.to(device)
        t1 = time.time()
        train_outputs = model(train_pep_inputs, train_pep_lens,device)
        train_loss = criterion(train_outputs, train_labels)
        time_train_ep += time.time() - t1

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        y_true_train = train_labels.cpu().numpy()
        y_prob_train = nn.Softmax(dim = 1)(train_outputs)[:, 1].cpu().detach().numpy()
        
        y_true_train_list.extend(y_true_train)
        y_prob_train_list.extend(y_prob_train)
        loss_train_list.append(train_loss.item())
       
    y_pred_train_list = transfer(y_prob_train_list, threshold)
    
    ys_train = (y_true_train_list, y_pred_train_list, y_prob_train_list)
    
    print('Fold-{}****Train (Ep avg): Epoch-{}/{} | Loss = {:.4f} | Time = {:.4f} sec'.format(fold, epoch, epochs, f_mean(loss_train_list), time_train_ep))
    
    y_true_train_list = np.array(y_true_train_list)
    y_pred_train_list = np.array(y_pred_train_list)
    
    metrics_train = performances(y_true_train_list, y_pred_train_list)
    return ys_train, loss_train_list, metrics_train, time_train_ep
              



def eval_step(model, val_loader, fold, epoch, epochs, criterion, threshold, device):
    # device = torch.device("cuda" if use_cuda else "cpu")
    
    model.eval()
    torch.manual_seed(2022)
    torch.cuda.manual_seed(2022)
    with torch.no_grad():
        loss_val_list = []
        y_true_val_list, y_prob_val_list = [], []
        for val_pep_inputs, val_pep_lens, val_labels in tqdm(val_loader):
            val_pep_inputs, val_labels = val_pep_inputs.to(device), val_labels.to(device)
            val_pep_lens = val_pep_lens.to(device)
            val_outputs = model(val_pep_inputs, val_pep_lens,device)
            val_loss = criterion(val_outputs, val_labels)

            y_true_val = val_labels.cpu().numpy()
            y_prob_val = nn.Softmax(dim = 1)(val_outputs)[:, 1].cpu().detach().numpy()

            y_true_val_list.extend(y_true_val)
            y_prob_val_list.extend(y_prob_val)
            loss_val_list.append(val_loss.item())

        y_pred_val_list = transfer(y_prob_val_list, threshold)
        
        y_true_val_list = np.array(y_true_val_list) 
        y_pred_val_list = np.array(y_pred_val_list)
        y_prob_val_list = np.array(y_prob_val_list)
        # ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)

        print('Fold-{} ****Test  Epoch-{}/{}: Loss = {:.6f}'.format(fold, epoch, epochs, f_mean(loss_val_list)))
        
        TP,FP,FN,TN,fpr,tpr,auc, aupr,f1_score, accuracy, recall, specificity, precision =  performances(y_true_val_list, y_prob_val_list)         
    return TP,FP,FN,TN,fpr,tpr,auc, aupr,f1_score, accuracy, recall, specificity, precision



#G:TransPHLA-AOMP/Dataset/
def data_with_loader(type_ = 'train',fold = None,  batch_size = 128):
    if type_ != 'train' and type_ != 'val':
        data = pd.read_csv('/home_exp_2/jiani.ma/mhc/comparison-experiments/STMHCPan/data/5fold/fold{}/test_data_fold{}.csv'.format(fold,fold))
    elif type_ == 'train':
        data = pd.read_csv('/home_exp_2/jiani.ma/mhc/comparison-experiments/STMHCPan/data/5fold/fold{}/train_data_fold{}.csv'.format(fold,fold)) #, index_col = 0
    else:  
        data = pd.read_csv('/home_exp_2/jiani.ma/mhc/comparison-experiments/STMHCPan/data/5fold/fold{}/valid_data_fold{}.csv'.format(fold,fold)) #, index_col = 0 
    pep_inputs, hla_inputs, labels, pep_lens = make_data(data, pep_max_len, hla_max_len)
    loader = Data.DataLoader(MyDataSet(pep_inputs, hla_inputs, pep_lens, labels), batch_size, shuffle = False, num_workers = 0)
    return data, pep_inputs, hla_inputs, pep_lens, labels, loader



if __name__ == "__main__":
    hla_sequence = pd.read_csv('/home_exp_2/jiani.ma/mhc/comparison-experiments/STMHCPan/data/HLAI_pseudosequences_34mer.csv')
    pep_max_len = 15 # peptide; enc_input max sequence length
    hla_max_len = 34 # hla; dec_input(=dec_output) max sequence length
    tgt_len = pep_max_len + hla_max_len   # 49

    vocab = {'-': 0, 'Y': 1, 'A': 2, 'T': 3, 'V': 4, 'L': 5, 'D': 6, 'E': 7, 'G': 8, 'R': 9, 'H': 10, 'I': 11, 'W': 12, 'Q': 13, 'K': 14, 'M': 15, 'F': 16, 'N': 17, 'S': 18, 'P': 19, 'C': 20, 'X': 21}

    vocab_size = len(vocab)

    n_layers = 1  # number of Encoder of Decoder Layer
    n_heads = 8

    batch_size = 1024
    epochs = 30
    threshold = 0.5

    device = "cpu" 

    model_place = "/home_exp_2/jiani.ma/mhc/comparison-experiments/STMHCPan/model_param_dict/model_dict_for_fold_1.pkl"
    checkpoint = torch.load(model_place)
    model = STSeqCls((vocab_size, 100), num_cls=2, hidden_size=300, num_layers=1, num_head=8, max_len=49,cls_hidden_size=600,dropout=0.1,head_dim=32).to(device)
    model.load_state_dict(checkpoint)


    total_labels = []
    total_features = [] 
    
    for fold in range(5):
        print('=====Fold-{}====='.format(fold))
        print('-----Generate data loader-----')
        test_data, test_pep_inputs, test_hla_inputs, test_pep_lens, test_labels, test_loader = data_with_loader(type_ = 'test', fold = fold,  batch_size = batch_size)
        with torch.no_grad():      
            for val_pep_inputs, val_pep_lens, val_labels in tqdm(test_loader):
                val_pep_inputs, val_labels = val_pep_inputs.to(device), val_labels.to(device)
                val_pep_lens = val_pep_lens.to(device)
                features = model(val_pep_inputs, val_pep_lens,device)
                y_true_val = val_labels.cpu().numpy()
                total_labels.extend(y_true_val)
                total_features.extend(features)
    
    total_labels = np.array(total_labels) 
    total_features = np.array(total_features)
    

    feature_mat_df = pd.DataFrame(total_features)
    feature_mat_df.to_excel("/home_exp_2/jiani.ma/mhc/comparison-experiments/STMHCPan/TSNE/sample_features.xlsx")

    labels_df = pd.DataFrame(total_labels) 
    labels_df.to_excel("/home_exp_2/jiani.ma/mhc/comparison-experiments/STMHCPan/TSNE/labels.xlsx")
        
        
        
        
        

