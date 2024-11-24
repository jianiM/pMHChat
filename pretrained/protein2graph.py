"""
@author: amber
"""
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd 
import torch 
from torch_geometric.data import Dataset, Data,InMemoryDataset
from utils import cmap2graph 
from config_init import get_config

class ProteinGraphDataset(InMemoryDataset):
    def __init__(self,root='/tmp',transform=None,pre_transform=None,sample_list=None,mode=None,ratio=None,peptide_info_dict_dir=None,mhc_info_dict_dir=None):
        super(ProteinGraphDataset,self).__init__(root,transform,pre_transform)
        self.mode = mode
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(sample_list,ratio,peptide_info_dict_dir,mhc_info_dict_dir)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property 
    def raw_file_names(self):
        pass
    
    @property
    def processed_file_names(self):
        return [self.mode+'.pt']

    def download(self):
        pass 
    
    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)


    def process(self,sample_list,ratio,peptide_info_dict_dir,mhc_info_dict_dir):
        data_list = [] 
        data_len = len(sample_list)
        for i in range(data_len):
            print('Converting peptide to graph: {}/{}'.format(i+1, data_len))
            pair = sample_list.iloc[i]
            node_features,edge_index,mhc_residue_features,target = self._get_geometric_input(pair,ratio,peptide_info_dict_dir,mhc_info_dict_dir)
            GCNData = Data(x=node_features, edge_index=torch.LongTensor(edge_index),y=torch.FloatTensor([target]))
            GCNData.mhc_embed = torch.FloatTensor(mhc_residue_features) 
            data_list.append(GCNData)
        print("Graph construction done. Saving to file.")
        # print("data list for pytorch geometric:",data_list)         
        if self.pre_filter is not None:
           data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
           data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


    def _get_geometric_input(self,pair,ratio,peptide_info_dict_dir,mhc_info_dict_dir):
        peptide = pair['peptide']
        mhc = pair['mhc'] 
        target = pair['label'] # af = pair['af']
        peptide_file = peptide +'.pt' 
        peptide_file_path = os.path.join(peptide_info_dict_dir,peptide_file)
        peptide_info = torch.load(peptide_file_path)                        # peptide_info: feature_representation + cmap 
        mhc_file = mhc + '.pt' 
        mhc_file_path = os.path.join(mhc_info_dict_dir,mhc_file)     
        mhc_info = torch.load(mhc_file_path)                                # mhc_info: feature_representation
        node_features = peptide_info['feature_representation']   # node feature         
        peptide_cmap = peptide_info['cmap']                                 # node linkage 
        edge_index = cmap2graph(contact_map=peptide_cmap,ratio=ratio)
        mhc_residue_features = mhc_info['feature_representation']           # additional information   
        if len(mhc_residue_features.shape) == 2:
            mhc_residue_features = mhc_residue_features.unsqueeze(0)        # Add a batch dimension
        return node_features,edge_index,mhc_residue_features,target


if __name__ == "__main__": 
    config = get_config() 
    root_path = config.root_path_topofallfeature
    dataset = config.dataset_topofallfeature
    project = config.project_topofallfeature
    
    kfold = config.kfold_topofallfeature
    ratio = config.ratio_topofallfeature
    fold_root_path = config.fold_root_path_topofallfeature        # "/home_exp_2/jiani.ma/mhc/NetMHCIIpan-3.2/5fold/"

    peptide_info_dict_dir = config.peptide_info_dict_dir_topofallfeature
    mhc_info_dict_dir = config.mhc_info_dict_dir_topofallfeature
    peptide_info_dict_dir = os.path.join(root_path,project,peptide_info_dict_dir) 
    mhc_info_dict_dir = os.path.join(root_path,project,mhc_info_dict_dir) 


    for i in range(kfold):
        print("This is fold:",i+1)
        print('-'*20)
        
        fold_path = os.path.join(fold_root_path,'fold'+str(i+1))            # "/home_exp_2/jiani.ma/mhc/pmhchat-pretrained/NetMHCIIpan-3.2-pretrained/5fold/"

        train_list_file = f"train{i+1}"+'.txt'  
        test_list_file = f"test{i+1}"+'.txt'   
    
        # generate balanced dataset 
        train_info = pd.read_csv(os.path.join(fold_path,'raw',train_list_file),sep='\t')   # orig train data 
        test_info = pd.read_csv(os.path.join(fold_path,'raw',test_list_file),sep='\t')     # orig test data 
        
        train_data = ProteinGraphDataset(root=fold_path,sample_list=train_info,mode="train",ratio=ratio,peptide_info_dict_dir=peptide_info_dict_dir,mhc_info_dict_dir=mhc_info_dict_dir)
        test_data = ProteinGraphDataset(root=fold_path,sample_list=test_info,mode="test",ratio=ratio,peptide_info_dict_dir=peptide_info_dict_dir,mhc_info_dict_dir=mhc_info_dict_dir)
        
        
       
       
