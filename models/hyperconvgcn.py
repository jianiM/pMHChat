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
        self.pep_fc = nn.Linear(320,128)
    
        self.fc1 = nn.Linear(34*24, 16)
        self.out = nn.Linear(16, n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)


    def forward(self,data):
        x, edge_index, mhc_embed, batch = data.x, data.edge_index, data.mhc_embed, data.batch 
        
        # mhc process 
        mhc_out,(_, _) = self.bilstm_layer(mhc_embed)           # mhc_outputs ([bs, 34, 128])

        
        # peptide process 
        x = self.hyperconv1(x, edge_index)
        x = self.relu(x)
        x = self.hyperconv2(x, edge_index)
    
        pep_out = x.view(mhc_out.size(0), 24, 320)            # [bs,15,320]
        pep_out =self.pep_fc(pep_out)                   # [bs,15,128]
       
        clus_map = torch.bmm(mhc_out, pep_out.permute(0, 2, 1))  # ([bs, 34, 15])

        xc = clus_map.view(clus_map.size(0),-1)     # ([bs, 34*15]) 
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        
        return out,clus_map









