import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import HypergraphConv
from torch_geometric.nn import global_max_pool as gmp



class HyperConvNet(nn.Module):
    def __init__(self,mhc_dim,mhc_hidden,peptide_dim,drop_prob,n_output):
        super(HyperConvNet, self).__init__()
        self.bilstm_layer = nn.LSTM(input_size = mhc_dim,hidden_size = mhc_hidden,num_layers = 2,batch_first = True,dropout = drop_prob,bidirectional = True)
        self.hyperconv1 = HypergraphConv(in_channels=peptide_dim, out_channels=peptide_dim)
        self.hyperconv2 = HypergraphConv(in_channels=peptide_dim, out_channels=peptide_dim)
        self.fc1 = nn.Linear(320+34*128, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)


    def forward(self,data):
        x, edge_index, mhc_embed, batch = data.x, data.edge_index, data.mhc_embed, data.batch 
        
        # mhc process
        mhc_outputs,(_, _) = self.bilstm_layer(mhc_embed)           # mhc_outputs ([8, 34, 256])
        mhc_out = mhc_outputs.reshape(mhc_outputs.size(0), -1)      # print('mhc_out:',mhc_out.size())  # why view not work
        
        # peptide process 
        x = self.hyperconv1(x, edge_index)
        x = self.relu(x)
        x = self.hyperconv2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch) 
        
        xc = torch.cat((mhc_out,x),dim=1)   # torch.Size([8, 9024]
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc) 
        return out









