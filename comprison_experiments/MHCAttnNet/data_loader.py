# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:00:11 2024

@author: amber
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from protvec_coding import split_ngrams,generate_corpusfile,MHCProtVec,PeptideProtVec


# mhcvec = MHCProtVec()
# pepvec = PeptideProtVec()  


class MHCpeptideDataset(Dataset):
    def __init__(self, info, mhc_df, peptide_df,mhcvec,pepvec,mhc_weight,pep_weight,task): 

        super(MHCpeptideDataset,self).__init__()
        self.info = info 
        self.mhc_df = mhc_df 
        self.peptide_df = peptide_df 
        self.task = task 

        self.mhcvec = mhcvec 
        self.mhc_weight = mhc_weight
        # self.mhc_weight = self.mhcvec.load_protvec(model_weights='/home_exp_2/jiani.ma/mhc/comparison-experiments/MHCAttenNet/dataset/1-gram-model-weights.mdl')  
        self.mhc_vocab = self.mhcvec.m.wv.index_to_key   

        self.pepvec = pepvec
        self.pep_weight = pep_weight
        # self.pep_weight = self.pepvec.load_protvec(model_weights='/home_exp_2/jiani.ma/mhc/comparison-experiments/MHCAttenNet/dataset/3-gram-model-weights.mdl')  
        self.peptide_vocab = self.pepvec.m.wv.index_to_key 


    def __getitem__(self, idx):
        pair = self.info.iloc[idx] 
        peptide = pair['peptide'] 
        mhc = pair['mhc'] 
        if self.task == "regression": 
            target = pair['af']     
        else:
            target = pair['label'] 

        mhc_seq = self.mhc_df.loc[mhc, 'pseudosequence']
        peptide_seq = self.peptide_df.loc[peptide,'Seq']
        
        # tokenize the mhc sequence 
        splitted_mhc_seq = split_ngrams(mhc_seq,1)[0] 
        valid_mhcgrams = [ngram for ngram in splitted_mhc_seq if ngram in self.mhc_vocab]  
        mhc_indices = [self.mhcvec.m.wv.key_to_index[ngram] for ngram in valid_mhcgrams]  

        # tokenize the peptide sequence 
        splitted_pep_seq = split_ngrams(peptide_seq,3)[0] 
        valid_pepgrams = [ngram for ngram in splitted_pep_seq if ngram in self.peptide_vocab]
        pep_indices = [self.pepvec.m.wv.key_to_index[ngram] for ngram in valid_pepgrams]   
        return mhc_indices,pep_indices,target

    def __len__(self):
            return len(self.info)
    


def collate_fn(batch,task):
    mhc_indices,pep_indices,target = list(zip(*batch))      # feature has already been tensor format.
    mhc_indices = torch.LongTensor(mhc_indices)
    pep_indices = torch.LongTensor(pep_indices)
    
    if task == "regression": 
        target = torch.FloatTensor(target)
    else:
        target = torch.LongTensor(target) 
    return mhc_indices, pep_indices, target












# class MHCpeptideDataset(Dataset):
#     def __init__(self, info, mhc_embedding_dict_path, peptide_embedding_dict_path):
#         super(MHCpeptideDataset,self).__init__()
#         self.info = info
#         self.mhc_embedding_dict = torch.load(mhc_embedding_dict_path)
#         self.peptide_embedding_dict = torch.load(peptide_embedding_dict_path) 


#     def __getitem__(self, idx):
#         pair = self.info.iloc[idx] 
#         peptide = pair['peptide']
#         mhc = pair['mhc']
#         label = pair['label']
#         mhc_embed = self.mhc_embedding_dict[mhc] 
#         peptide_embed = self.peptide_embedding_dict[peptide] 
#         return peptide_embed,mhc_embed,label 
    
#     def __len__(self):
#         return len(self.info)
 

# def collate_fn(batch):  
#     peptide_embeds = [item[0] for item in batch]  
#     mhc_embeds = [item[1] for item in batch]  
#     labels = [item[2] for item in batch]  
#     peptide_embed_batch = torch.stack(peptide_embeds, dim=0)  
#     mhc_embed_batch = torch.stack(mhc_embeds, dim=0)  
#     labels_tensor = torch.tensor(labels, dtype=torch.long)  
#     return peptide_embed_batch, mhc_embed_batch, labels_tensor
