import argparse
def get_config():
    
    parse = argparse.ArgumentParser(description='common train config')  
    
    parse.add_argument('-root_path', '--root_path_topofallfeature', type=str, default="/home_exp_2/jiani.ma/mhc/",help = "root path for this project")
    parse.add_argument('-dataset', '--dataset_topofallfeature', type=str, default="BD2016", help="dataset") 
    parse.add_argument('-project', '--project_topofallfeature', type=str, default="pmhchat/", help="project name") 
    parse.add_argument('-task', '--task_topofallfeature', type=str, default="regression", help="regression or classification") 

    # create_peptide_features.py 
    parse.add_argument('-peptide_df_path', '--peptide_df_path_topofallfeature', type=str, default="peptide_df.xlsx",help="peptide information file for recording fasta seqs") 
    parse.add_argument('-peptide_info_dict_dir', '--peptide_info_dict_dir_topofallfeature', type=str, default="peptide_info_dict/", help="dict for storing peptide info: fastas and cmap") 
    parse.add_argument('-max_peptide_len', '--max_peptide_len_topofallfeature', type=int, default=15) 
    parse.add_argument('-padding', '--padding_topofallfeature', type=str, default='<pad>') 
    
    # create_mhc_features.py
    parse.add_argument('-mhc_df_path', '--mhc_df_path_topofallfeature', type=str, default="mhc_df.xlsx",help="mhc information file for recording pseudosequences") 
    parse.add_argument('-mhc_info_dict_dir', '--mhc_info_dict_dir_topofallfeature', type=str, default="mhc_info_dict/", help="dict for storing mhc info: mhc pseudosequences") 

    # protein2graph and main
    parse.add_argument('-kfold', '--kfold_topofallfeature', type=int, default=5)
    parse.add_argument('-ratio', '--ratio_topofallfeature', type=float, default=0.5, help="building the peptide graph") 
    parse.add_argument('-fold_root_path', '--fold_root_path_topofallfeature', type=str, default="/home_exp_2/jiani.ma/mhc/BD2016/5fold/")
    parse.add_argument('-modelling', '--modelling_topofallfeature', type=str, nargs='?', default= "hyperconv", help="gnn model choice") 
    parse.add_argument('-cuda_name', '--cuda_name_topofallfeature', type=str, default="cuda:0") 
    parse.add_argument('-lr', '--lr_topofallfeature', type=float, default=1e-4,help="learning rate") 
    parse.add_argument('-weight_decay', '--weight_decay_topofallfeature', type=float, default=5e-4)
    parse.add_argument('-num_epoches', '--num_epoches_topofallfeature', type=int, default=100)  
    parse.add_argument('-train_batch_size', '--train_batch_size_topofallfeature', type=int, default=256) 
    parse.add_argument('-test_batch_size', '--test_batch_size_topofallfeature', type=int, default=8)
    parse.add_argument('-model_saving_dir', '--model_saving_dir_topofallfeature', type=str, default="/home_exp_2/jiani.ma/mhc/pmhchat/model_saving/")
    config = parse.parse_args()
    return config