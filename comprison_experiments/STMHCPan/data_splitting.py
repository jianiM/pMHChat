import os
import pandas as pd
from sklearn.model_selection import train_test_split

random_seed = 42


kfold = 5 

fold_root_path = "/home_exp_2/jiani.ma/mhc/comparison-experiments/STMHCPan/data/5fold/"

pseudosequences_df_path = "/home_exp_2/jiani.ma/mhc/LOMO/STMHCPan/data/HLAI_pseudosequences_34mer.csv"
pseudosequences_df = pd.read_csv(pseudosequences_df_path,index_col=0)

peptide_df_path = "/home_exp_2/jiani.ma/mhc/LOMO/STMHCPan/data/peptide_df.xlsx"
peptide_df = pd.read_excel(peptide_df_path)

peptide_df = peptide_df.rename(columns={'Peptide': 'Peptide_Name'})  


for i in range(kfold):
    
    fold_path = os.path.join(fold_root_path,'fold'+str(i))

    train_file = f"train"+'.txt'  
    test_file = f"test"+'.txt'  

    train_path = os.path.join(fold_path,'raw',train_file)
    test_path = os.path.join(fold_path,'raw',test_file) 

    train_data = pd.read_csv(train_path,sep='\t') 
    test_data = pd.read_csv(test_path,sep='\t') 
    
    train_data = train_data.rename(columns={'peptide': 'pid'})  
    test_data = test_data.rename(columns={'peptide': 'pid'})  
    
    train_data = train_data.merge(pseudosequences_df, left_on='mhc', right_on='allele', how='left')
    test_data = test_data.merge(pseudosequences_df, left_on='mhc', right_on='allele', how='left')


    train_data = train_data.rename(columns={'pseudosequence': 'HLA_sequence'})  
    test_data = test_data.rename(columns={'pseudosequence': 'HLA_sequence'})  


    train_data = train_data.merge(peptide_df, left_on='pid', right_on='Peptide_Name', how='left')
    test_data = test_data.merge(peptide_df, left_on='pid', right_on='Peptide_Name', how='left')

    train_data_split, valid_data_split = train_test_split(train_data, test_size=0.2, random_state=random_seed, stratify=train_data['label'])


    train_data_split.to_csv(os.path.join(fold_path, f"train_data_fold{i}.csv"), index=False)
    valid_data_split.to_csv(os.path.join(fold_path, f"valid_data_fold{i}.csv"), index=False)
    test_data.to_csv(os.path.join(fold_path, f"test_data_fold{i}.csv"), index=False)













