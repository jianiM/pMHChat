# model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm_msa1b_t12_100M_UR50S")

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:09:54 2024
@author: amber
use esmMSA1b to extract the features of mhc, each pseudosequence of mhc was processed as a dict, which includes residue-level features for 
multi-head attention network
# model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm_msa1b_t12_100M_UR50S")
"""

import torch
import esm
import pandas as pd
import os
import torch.nn as nn
from data_loader import MHCpeptideDataset,collate_fn
from torch.utils.data import DataLoader 
from model import FineTuneESMMSA1B,FineTuneESM2,PreTrain
from utils import trimming_fasta,get_metric

torch.cuda.manual_seed(1029)
torch.manual_seed(1029)


def training(model,train_loader,device,optimizer,loss_fn):            
    epoch_loss = 0.0 
    train_num = 0.0 
    model.train()
    for batch_idx, (batch_train_peptide, batch_train_mhc, batch_y) in enumerate(train_loader):     
        batch_y = batch_y.view(-1).float().to(device)
        optimizer.zero_grad() 
        out = model(batch_train_mhc,batch_train_peptide).view(-1)
        loss = loss_fn(out,batch_y) 
        loss.backward()
        optimizer.step()  
        train_num += len(batch_train_peptide)
        epoch_loss += loss.item() * len(batch_train_peptide)
    return epoch_loss/train_num

    

def predicting(test_loader,model,device):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for test_idx, (batch_test_peptide, batch_test_mhc, batch_test_y) in enumerate(test_loader):            
            output = model(batch_test_mhc,batch_test_peptide)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, batch_test_y.view(-1, 1).cpu()), 0)
        total_labels_arr = total_labels.numpy().flatten()
        total_preds_arr = total_preds.numpy().flatten() 
    TP,FP,FN,TN,fpr,tpr,auc, aupr,f1_score, accuracy, recall, specificity, precision = get_metric(total_labels_arr, total_preds_arr)
    return TP,FP,FN,TN,fpr,tpr,auc, aupr,f1_score, accuracy, recall, specificity, precision



if __name__ == "__main__": 
    
    kfold = 5     
    max_peptide_len = 15
    padding = '<pad>'
    cuda_name = "cuda:0"
    n_output = 1 
    drop_prob = 0.5 
    lr = 1e-4
    weight_decay = 5e-4
    num_epoches = 50
    train_batch_size = 128
    test_batch_size = 8 
    
    mhc_df_path = "/home_exp_2/jiani.ma/mhc/NetMHCIIpan-3.2/mhc_df.xlsx"
    peptide_df_path = "/home_exp_2/jiani.ma/mhc/NetMHCIIpan-3.2/peptide_df.xlsx"
    fold_root_path = "/home_exp_2/jiani.ma/mhc/NetMHCIIpan-3.2/5fold"
    model_saving_dir = "/home_exp_2/jiani.ma/mhc/llm_pretrain/model_saving/"
    
    
    # mhc sequence archive
    mhc_df = pd.read_excel(mhc_df_path)  
    mhc_ids = mhc_df['mhc-allele'].tolist()
    mhc_seqs = mhc_df['pseudosequence'].tolist()
    mhc_df = pd.DataFrame({'pseudosequence': mhc_seqs}, index=mhc_ids)
    
    
    # peptide sequence archive 
    peptide_df = pd.read_excel(peptide_df_path)  
    peptide_ids = peptide_df['Peptide'].tolist()  # will be in use later 
    seqs = peptide_df['Seq'].tolist()
    trimmed_seqs = trimming_fasta(seqs,max_peptide_len,padding)
    peptide_df = pd.DataFrame({'Seq': trimmed_seqs}, index=peptide_ids)

    # Load the originial ESM-MSA-1b model that is waiting for fine-tuning mhc pseudoseqs
    mhc_model, mhc_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    peptide_model, peptide_alphabet = esm.pretrained.esm2_t6_8M_UR50D()

    
    for i in range(1):        #(kfold):
        print("This is fold:",i+1)
        print('-'*20)
        fold_path = os.path.join(fold_root_path,'fold'+str(i+1))
    
        train_list_file = f"train{i+1}"+'.txt'  
        test_list_file = f"test{i+1}"+'.txt'   
        model_file_name = 'model_dict_for_cpu.pkl' 
        
        #model_file_name = 'model_dict_for_fold_{}.pkl'.format(i+1)
        model_saving_path = os.path.join(model_saving_dir,model_file_name)
        
        train_info = pd.read_csv(os.path.join(fold_path,'raw',train_list_file),sep='\t')   # orig train data 
        test_info = pd.read_csv(os.path.join(fold_path,'raw',test_list_file),sep='\t')     # orig test data 
        
        train_data = MHCpeptideDataset(info=train_info, peptide_df=peptide_df, mhc_df=mhc_df) 
        test_data = MHCpeptideDataset(info=test_info, peptide_df=peptide_df, mhc_df=mhc_df) 
        
        
        train_size = int(0.8 * len(train_data))
        valid_size = len(train_data) - train_size
        train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])   
        
        train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True,collate_fn=collate_fn)
        valid_loader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=False,collate_fn=collate_fn)
        test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False,collate_fn=collate_fn)

        # example for extracting the batch data 
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")     
        model = PreTrain(mhc_model=mhc_model, mhc_alphabet=mhc_alphabet,peptide_model=peptide_model, peptide_alphabet=peptide_alphabet,n_output=n_output,drop_prob=drop_prob,device=device).to(device) 
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
        loss_fn = nn.BCEWithLogitsLoss().to(device)   
        best_val_auc = 0.0 

        for epoch in range(num_epoches):
            print('Epoch{}/{}'.format(epoch,(num_epoches-1)))
            print('*'*10)
            #### training procedure 
            epoch_loss = training(model,train_loader,device,optimizer,loss_fn)
            print("epoch_loss:",epoch_loss)
                       
            #### validating the model     
            val_TP,val_FP,val_FN,val_TN,val_fpr,val_tpr,val_auc, val_aupr,val_f1_score, val_accuracy, val_recall, val_specificity, val_precision = predicting(valid_loader,model,device) 
            
            # taking the aupr as golden standard, save the model parameter for loading in test
            if val_auc > best_val_auc: 
                #save the model parameter for loading in test
                print("val_auc:",val_auc)
                print('val_aupr:',val_aupr)
                torch.save(model, model_saving_path)
                best_val_auc = val_auc
                
        checkpoint = torch.load(model_saving_path)
        TP,FP,FN,TN,fpr,tpr,auc,aupr,f1_score, accuracy, recall, specificity, precision = predicting(test_loader,checkpoint,device)  
        print('test_auc:',auc)
        print('test_aupr:',aupr)
        print('f1_score:',f1_score)
        print('accuracy:',accuracy)
        print('recall:',recall)
        print('specificity:',specificity)
        print('precision:',precision)
                
        

    

    
    
    
    
    
 
    
    
    


    
    
    

