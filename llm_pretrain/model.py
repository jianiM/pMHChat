# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 18:06:57 2024

@author: amber
"""

import torch
import esm
import pandas as pd
import os
import torch.nn as nn


class FineTuneESMMSA1B(nn.Module):
    def __init__(self,mhc_model,mhc_alphabet,device):
        super(FineTuneESMMSA1B, self).__init__()
        self.device = device
        self.esmmsa = mhc_model.to(self.device)
        self.mhc_batch_converter = mhc_alphabet.get_batch_converter()
        # Freeze all layers except the final linear layer for later fine-tune 
        for param in self.esmmsa.parameters():
            param.requires_grad = False
        for param in self.esmmsa.lm_head.parameters():
            param.requires_grad = True 

    def forward(self,batch_mhcs):
        _, _, token = self.mhc_batch_converter(batch_mhcs)  
        token = token.to(self.device)
        mhc_results = self.esmmsa(token, repr_layers=[12])  
        mhc_representations = mhc_results['representations'][12].squeeze(0)[:,1:,:]   # [128,35,768]-->128:batch_size, 35:len. 768:dim 
        return mhc_representations
    

class FineTuneESM2(nn.Module):
    def __init__(self,peptide_model,peptide_alphabet,device):
        super(FineTuneESM2,self).__init__()
        self.device = device
        self.esm2 = peptide_model.to(self.device) 
        self.peptide_batch_converter = peptide_alphabet.get_batch_converter()
        # Freeze all layers except the final linear layer for later fine-tune 
        for param in self.esm2.parameters():
            param.requires_grad = False
        for param in self.esm2.lm_head.parameters():
            param.requires_grad = True 
        
    def forward(self,batch_peptides):
        _, _,tokens = self.peptide_batch_converter(batch_peptides) 
        tokens = tokens.to(self.device)
        peptide_results = self.esm2(tokens, repr_layers=[5], return_contacts=True)
        peptide_representations = peptide_results['representations'][5][:,1:-1,:] 
        contact_map = peptide_results['contacts']
        return peptide_representations,contact_map  
        
    

class PreTrain(nn.Module):
    def __init__(self,mhc_model, mhc_alphabet,peptide_model, peptide_alphabet,n_output,drop_prob,device):
        super(PreTrain,self).__init__() 
        
        self.esmmsa_model = FineTuneESMMSA1B(mhc_model, mhc_alphabet,device) 
        self.esm2_model = FineTuneESM2(peptide_model, peptide_alphabet,device)
        
        self.fc1 = nn.Linear(34*768+15*320,1024) 
        self.fc2 = nn.Linear(1024,128)
        self.out = nn.Linear(128,n_output) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
        
    
    def forward(self,batch_mhcs,batch_peptides): 
        
        mhc_representations = self.esmmsa_model(batch_mhcs)
        peptide_representations,contact_map = self.esm2_model(batch_peptides) 

        x_mhc = mhc_representations.reshape(mhc_representations.size(0),-1)
        x_peptide = peptide_representations.reshape(peptide_representations.size(0),-1)
        
        concate_features = torch.cat((x_mhc,x_peptide),dim=1)
        
        x = self.fc1(concate_features)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x) 
        x = self.dropout(x)
        
        output = self.out(x) 
        return output 
