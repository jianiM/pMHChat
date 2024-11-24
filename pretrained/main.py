import pandas as pd 
import numpy as np 
import os 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
from protein2graph import ProteinGraphDataset
import pickle
import torch
import torch.nn as nn 
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim
from config_init import get_config
from utils import cmap2graph, get_regression_metric, get_classfication_metric
import matplotlib.pyplot as plt
torch.cuda.manual_seed(1112)
torch.manual_seed(1112)
from torch_geometric.data import Dataset, Data,InMemoryDataset
from models.hyperconvgcn import HyperConvNet



def training(task,gnn_model,train_loader,device,optimizer,loss_fn):            
    epoch_loss = 0.0 
    train_num = 0.0 
    gnn_model.train()
    for train_idx, batch_train_data in enumerate(train_loader):
        batch_train_data = batch_train_data.to(device)
        if task == 'regression':
            batch_y = batch_train_data.y.view(-1).float()
        else: 
            batch_y = batch_train_data.y.view(-1).float()
            batch_y = torch.where(batch_y < 0.426, torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))  # true labels 
        optimizer.zero_grad() 
        out = gnn_model(batch_train_data).view(-1)
        loss = loss_fn(out,batch_y) 
        loss.backward()
        optimizer.step()  
        train_num += batch_train_data.size(0)
        epoch_loss += loss.item() * batch_train_data.size(0)
    return epoch_loss/train_num


def predicting_classification(test_loader,gnn_model,device):
    gnn_model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for test_idx, batch_test_data in enumerate(test_loader):
            batch_test_data = batch_test_data.to(device) 
            output = gnn_model(batch_test_data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, batch_test_data.y.view(-1, 1).cpu()), 0)
        total_labels_arr = total_labels.numpy().flatten()
        total_preds_arr = total_preds.numpy().flatten() 
        total_labels_arr_binary = (total_labels_arr >= 0.426).astype(float)  
    TP,FP,FN,TN,fpr,tpr,auc, aupr,f1_score, accuracy, recall, specificity, precision = get_classfication_metric(total_labels_arr_binary, total_preds_arr)
    return TP,FP,FN,TN,fpr,tpr,auc, aupr,f1_score, accuracy, recall, specificity, precision


def predicting_regression(test_loader,gnn_model,device):
    gnn_model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for test_idx, batch_test_data in enumerate(test_loader):
            batch_test_data = batch_test_data.to(device) 
            output = gnn_model(batch_test_data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, batch_test_data.y.view(-1, 1).cpu()), 0)
        total_labels_arr = total_labels.numpy().flatten()
        total_preds_arr = total_preds.numpy().flatten() 
    pcc = get_regression_metric(total_labels_arr, total_preds_arr)
    return pcc


if __name__ == "__main__": 
    config = get_config() 
    root_path = config.root_path_topofallfeature
    dataset = config.dataset_topofallfeature
    project = config.project_topofallfeature
    task = config.task_topofallfeature 
    
    kfold = config.kfold_topofallfeature
    ratio = config.ratio_topofallfeature
    fold_root_path = config.fold_root_path_topofallfeature    # "/home_exp_2/jiani.ma/mhc/NetMHCIIpan-3.2/5fold/"
    cuda_name = config.cuda_name_topofallfeature
    lr = config.lr_topofallfeature
    weight_decay = config.weight_decay_topofallfeature
    num_epoches = config.num_epoches_topofallfeature 
    train_batch_size = config.train_batch_size_topofallfeature 
    test_batch_size = config.test_batch_size_topofallfeature
    model_saving_dir = config.model_saving_dir_topofallfeature
    
    modelling = config.modelling_topofallfeature
    peptide_info_dict_dir = config.peptide_info_dict_dir_topofallfeature
    mhc_info_dict_dir = config.mhc_info_dict_dir_topofallfeature
    peptide_info_dict_dir = os.path.join(root_path,project,peptide_info_dict_dir) 
    mhc_info_dict_dir = os.path.join(root_path,project,mhc_info_dict_dir) 

    test_auc = np.zeros(kfold)
    test_aupr = np.zeros(kfold)
    test_f1_score = np.zeros(kfold)
    test_accuracy = np.zeros(kfold)
    test_recall = np.zeros(kfold)
    test_specificity = np.zeros(kfold)
    test_precision = np.zeros(kfold)
    test_pcc = np.zeros(kfold) 

    
    print('task:',task)

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
        train_data = ProteinGraphDataset(root=fold_path,sample_list=train_info,mode="train",ratio=ratio,peptide_info_dict_dir=peptide_info_dict_dir,mhc_info_dict_dir=mhc_info_dict_dir)
        test_data = ProteinGraphDataset(root=fold_path,sample_list=test_info,mode="test",ratio=ratio,peptide_info_dict_dir=peptide_info_dict_dir,mhc_info_dict_dir=mhc_info_dict_dir)

        # train-valid splitting
        train_size = int(0.8 * len(train_data))
        valid_size = len(train_data) - train_size
        train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])        

        train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)  # the dataloader derived from torch geometric
        valid_loader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=False)            
        test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)  
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")     
        gnn_model = HyperConvNet(mhc_dim=768,mhc_hidden=64,peptide_dim=320,drop_prob=0.5,n_output=1).to(device)
        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=lr,weight_decay=weight_decay)
        loss_fn = nn.BCEWithLogitsLoss().to(device)   
 
        best_val_pcc = 0.0
        best_val_auc = 0.0 
        
        for epoch in range(num_epoches):
            print('Epoch{}/{}'.format(epoch,(num_epoches-1)))
            print('*'*10)
            #### training procedure 
            epoch_loss = training(task,gnn_model,train_loader,device,optimizer,loss_fn)
            print("epoch_loss:",epoch_loss)
            
            #### validating the model     
            if task == "regression":
                val_pcc = predicting_regression(valid_loader,gnn_model,device)
                if val_pcc > best_val_pcc: 
                    print('val_pcc:',val_pcc)
                    torch.save(gnn_model, model_saving_path)
                    best_val_pcc = val_pcc
            else:
                val_TP, val_FP, val_FN, val_TN, val_fpr, val_tpr, val_auc, val_aupr,val_f1_score, val_accuracy, val_recall, val_specificity, val_precision = predicting_classification(valid_loader,gnn_model,device) 
                if val_auc > best_val_auc:                    
                    print('val_auc:',val_auc)
                    print('val_aupr:',val_aupr)
                    torch.save(gnn_model, model_saving_path)
                    best_val_auc = val_auc 
        
        # ######  test procedure 
        checkpoint = torch.load(model_saving_path)
        if task == "regression":
            pcc = predicting_regression(test_loader,checkpoint,device) 
            test_pcc[i] = pcc 
            print('pcc:',pcc)
        else:     
            TP,FP,FN,TN,fpr,tpr,auc,aupr,f1_score, accuracy, recall, specificity, precision = predicting_classification(test_loader,checkpoint,device)        
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



            
        
