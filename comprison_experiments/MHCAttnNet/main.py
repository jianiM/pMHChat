import pandas as pd 
import numpy as np 
import os 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import torch
import torch.nn as nn 
from sklearn.model_selection import train_test_split
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_loader import MHCpeptideDataset,collate_fn
torch.cuda.manual_seed(1029)
torch.manual_seed(1029)
from models import MHCAttnNet
from utils import get_regression_metric,get_classfication_metric
from protvec_coding import split_ngrams,generate_corpusfile,MHCProtVec,PeptideProtVec
from functools import partial


def training(model,train_loader,device,optimizer,loss_fn):            
    epoch_loss = 0.0 
    train_num = 0.0 
    model.train()
    for train_idx, (mhc_indices,pep_indices,label) in enumerate(train_loader):
        mhc_indices = mhc_indices.to(device) 
        pep_indices = pep_indices.to(device) 
        label = label.to(device).view(-1).float()   #128
        optimizer.zero_grad()
        out = model(pep_indices,mhc_indices).view(-1)
        loss = loss_fn(out,label) 
        loss.backward()
        optimizer.step()  
        train_num += mhc_indices.size(0)
        epoch_loss += loss.item() * mhc_indices.size(0)
    return epoch_loss/train_num


def predicting_classification(model,test_loader,device):            
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad(): 
        for idx, (test_mhc_indices,test_pep_indices,test_labels) in enumerate(test_loader):
            test_mhc_indices = test_mhc_indices.to(device) 
            test_pep_indices = test_pep_indices.to(device) 
            test_labels = test_labels.to(device).view(-1).float()
            output = model(test_pep_indices,test_mhc_indices)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, test_labels.view(-1, 1).cpu()), 0)
        total_labels_arr = total_labels.numpy().flatten()
        total_preds_arr = total_preds.numpy().flatten() 
        total_labels_arr_binary = (total_labels_arr >= 0.426).astype(float)  
    TP,FP,FN,TN,fpr,tpr,auc, aupr,f1_score, accuracy, recall, specificity, precision = get_classfication_metric(total_labels_arr_binary, total_preds_arr)
    return TP,FP,FN,TN,fpr,tpr,auc, aupr,f1_score, accuracy, recall, specificity, precision


def predicting_regression(model,test_loader,device):            
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad(): 
        for idx, (test_mhc_indices,test_pep_indices,test_labels) in enumerate(test_loader):
            test_mhc_indices = test_mhc_indices.to(device) 
            test_pep_indices = test_pep_indices.to(device) 
            test_labels = test_labels.to(device).view(-1).float()
            output = model(test_pep_indices,test_mhc_indices)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, test_labels.view(-1, 1).cpu()), 0)
        total_labels_arr = total_labels.numpy().flatten()
        total_preds_arr = total_preds.numpy().flatten() 
        pcc= get_regression_metric(total_labels_arr, total_preds_arr)
    return pcc
     


if __name__ == "__main__": 
    
    fold_root_path = "/home_exp_2/jiani.ma/mhc/BD2016/5fold/"
    cuda_name = "cuda:0"
    kfold = 5 
    lr = 0.01
    weight_decay = 5e-4
    num_epoches = 30
    
    train_batch_size = 128 
    test_batch_size = 8 

    model_saving_dir = "/home_exp_2/jiani.ma/mhc/comparison-experiments/MHCAttenNet/model_saving/"
    mhc_df_path = "/home_exp_2/jiani.ma/mhc/BD2016/mhc_df.xlsx"
    peptide_df_path = "/home_exp_2/jiani.ma/mhc/comparison-experiments/MHCAttenNet/dataset/trimmed_peptides.xlsx"

    task = "classification"

    mhc_df = pd.read_excel(mhc_df_path, index_col='mhc-allele') 
    peptide_df = pd.read_excel(peptide_df_path, index_col='Peptide') 

    mhcvec = MHCProtVec()
    mhc_weight = mhcvec.load_protvec(model_weights='/home_exp_2/jiani.ma/mhc/comparison-experiments/MHCAttenNet/dataset/1-gram-model-weights.mdl')  
    pepvec = PeptideProtVec()
    pep_weight = pepvec.load_protvec(model_weights='/home_exp_2/jiani.ma/mhc/comparison-experiments/MHCAttenNet/dataset/3-gram-model-weights.mdl')  

    test_auc = np.zeros(kfold)
    test_aupr = np.zeros(kfold)
    test_f1_score = np.zeros(kfold)
    test_accuracy = np.zeros(kfold)
    test_recall = np.zeros(kfold)
    test_specificity = np.zeros(kfold)
    test_precision = np.zeros(kfold)
    test_pcc = np.zeros(kfold)


    for i in range(kfold):
        print("This is fold:",i+1)
        print('-'*20)
        fold_path = os.path.join(fold_root_path,'fold'+str(i+1))
    
        train_list_file = f"train{i+1}"+'.txt'  
        test_list_file = f"test{i+1}"+'.txt'   
    
        model_file_name = 'model_dict_for_{}_for_fold_{}.pkl'.format(task, i+1)
        model_saving_path = os.path.join(model_saving_dir,model_file_name)
        
        train_info = pd.read_csv(os.path.join(fold_path,'raw',train_list_file),sep='\t')   # orig train data 
        test_info = pd.read_csv(os.path.join(fold_path,'raw',test_list_file),sep='\t')     # orig test data 

        train_data = MHCpeptideDataset(train_info,mhc_df, peptide_df,mhcvec,pepvec,mhc_weight,pep_weight,task)
        test_data = MHCpeptideDataset(test_info,mhc_df, peptide_df,mhcvec,pepvec,mhc_weight,pep_weight,task) 
        
        train_size = int(0.8 * len(train_data))
        valid_size = len(train_data) - train_size
        train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])   
        
        train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True,collate_fn=partial(collate_fn, task=task))
        valid_loader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=False,collate_fn=partial(collate_fn, task=task))
        test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False,collate_fn=partial(collate_fn, task=task))

        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")     
        model = MHCAttnNet(mhcvec, pepvec, mhc_weight, pep_weight).to(device) 
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
        loss_fn = nn.BCEWithLogitsLoss().to(device)   
        

        #### training procedure
        best_val_pcc = 0.0
        best_val_auc = 0.0 
        
        for epoch in range(num_epoches):
            print('Epoch{}/{}'.format(epoch,(num_epoches-1)))
            print('*'*10)
            epoch_loss = training(model,train_loader,device,optimizer,loss_fn)
            print("epoch_loss:",epoch_loss)

            #### regression 
            if task == "regression": 
                val_pcc = predicting_regression(model,valid_loader,device)
                if val_pcc > best_val_pcc: 
                    print('val_pcc:',val_pcc)
                    torch.save(model, model_saving_path)
                    best_val_pcc = val_pcc 

            else: 
                val_TP,val_FP,val_FN,val_TN,val_fpr,val_tpr,val_auc, val_aupr,val_f1_score, val_accuracy, val_recall, val_specificity, val_precision = predicting_classification(model,valid_loader,device)
                if val_auc > best_val_auc: 
                    print("val_auc:",val_auc)
                    print('val_aupr:',val_aupr)
                    torch.save(model, model_saving_path)
                    best_val_auc = val_auc 
        

        #######  test procedure 
        checkpoint = torch.load(model_saving_path)
        if task == "regression":
            pcc = predicting_regression(checkpoint,test_loader,device) 
            test_pcc[i] = pcc 
            print('pcc:',pcc)
        
        else:     
            TP,FP,FN,TN,fpr,tpr,auc,aupr,f1_score, accuracy, recall, specificity, precision = predicting_classification(checkpoint,test_loader,device)        
            test_auc[i] = auc
            test_aupr[i] = aupr
            test_f1_score[i] = f1_score
            test_accuracy[i] = accuracy
            test_recall[i] = recall
            test_specificity[i] = specificity
            test_precision[i] = precision    

            print('TP:',TP)
            print('FP:',FP)
            print('FN:',FN)
            print('TN:',TN) 
            print('fpr:',fpr)
            print('tpr:',tpr)
            print('test_auc:',auc)
            print('test_aupr:',aupr)
            print('f1_score:',f1_score)
            print('accuracy:',accuracy)
            print('recall:',recall)
            print('specificity:',specificity)
            print('precision:',precision)

    if task == 'regression': 
        mean_pcc = np.mean(test_pcc)  
        std_pcc = np.std(test_pcc)
        print('mean_pcc:',mean_pcc) 
        print('std_pcc:',std_pcc) 
    else: 
        mean_auroc = np.mean(test_auc)
        mean_aupr = np.mean(test_aupr)
        mean_f1 = np.mean(test_f1_score)
        mean_acc = np.mean(test_accuracy)  
        mean_recall = np.mean(test_recall)
        mean_specificity = np.mean(test_specificity)
        mean_precision = np.mean(test_precision)
        print('mean_auroc:',mean_auroc)
        print('mean_aupr:',mean_aupr)
        print('mean_f1:',mean_f1)
        print('mean_acc:',mean_acc)
        print('mean_recall:',mean_recall)
        print('mean_specificity:',mean_specificity)
        print('mean_precision:',mean_precision)
        std_auc = np.std(test_auc)
        std_aupr = np.std(test_aupr)
        std_f1 = np.std(test_f1_score)
        std_acc = np.std(test_accuracy)
        std_recall = np.std(test_recall)
        std_specificity = np.std(test_specificity)
        std_precision = np.std(test_precision)
        print('std_auc:',std_auc)
        print('std_aupr:',std_aupr)
        print('std_f1:',std_f1)
        print('std_acc:',std_acc)
        print('std_recall:',std_recall)
        print('std_specificity:',std_specificity)
        print('std_precision:',std_precision)



            
        