import torch 
import esm
import pandas as pd 
import os

model_saving_path = "/home_exp_2/jiani.ma/mhc/llm_pretrain/model_saving/model_dict_for_cpu.pkl"

checkpoint = torch.load(model_saving_path)    #,map_location='cpu')


finetuned_mhc_model = checkpoint.esmmsa_model


mhc_df_path = "/home_exp_2/jiani.ma/mhc/pmhchat-pretrained/NetMHCIIpan-3.2-pretrained/mhc_df.xlsx"

#  read the peptide ids and seqs 
mhc_df = pd.read_excel(mhc_df_path)    
mhc_alles = mhc_df['mhc-allele'].tolist()  # will be in use later 
pseudoseqs = mhc_df['pseudosequence'].tolist()
data = list(zip(mhc_alles,pseudoseqs))   # 5628

mhc_info_dict_dir = "/home_exp_2/jiani.ma/mhc/pmhchat-pretrained/pretraiend_mhc_info_dict/"

for i in range(len(data)):
    mhc_info_dict = {}    
    item_form = []
    item = data[i] 
    item_form.append(item)    #[(),()]
    mhc = mhc_alles[i]
    finetuned_mhc_model.eval()
    representations = finetuned_mhc_model(item_form).detach().cpu()
    mhc_info_dict['mhc'] = mhc
    mhc_info_dict['feature_representation'] = representations.squeeze(0)
    file_name = mhc + ".pt"  
    file_path = os.path.join(mhc_info_dict_dir,file_name) 
    torch.save(mhc_info_dict,file_path) 
    print('Transformation for mhc {} is complete!'.format(mhc))

