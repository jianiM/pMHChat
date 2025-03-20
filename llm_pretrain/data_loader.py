# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:00:11 2024

@author: amber
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class MHCpeptideDataset(Dataset):
    def __init__(self, info,peptide_df,mhc_df):
        super(MHCpeptideDataset,self).__init__()
        self.info = info
        self.peptide_df = peptide_df 
        self.mhc_df = mhc_df    

    def __getitem__(self, idx):
        pair = self.info.iloc[idx]
        peptide = pair['peptide']
        mhc = pair['mhc']
        label = pair['label']
        peptide_seq = self.peptide_df.loc[peptide,'Seq']
        mhc_seq = self.mhc_df.loc[mhc,'pseudosequence'] 
        peptide_item = (peptide, peptide_seq)
        mhc_item = (mhc, mhc_seq)
        return peptide_item, mhc_item, label

    def __len__(self):
        return len(self.info)
    

    
def collate_fn(batch):
    peptide_items, mhc_items, labels = list(zip(*batch))      # feature has already been tensor format.
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return peptide_items, mhc_items, label_tensor
