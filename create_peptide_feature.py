import torch 
import esm
import pandas as pd 
import os

def trimming_fasta(fasta_seqs,trimmed_thresh,padding):
    trimmed_fastas = []
    for i in range(len(fasta_seqs)):
        fasta = fasta_seqs[i]
        seq_len = len(fasta)
        if seq_len > trimmed_thresh: 
            trimmed_fasta = fasta[0:trimmed_thresh]
        elif seq_len < trimmed_thresh:
            trimmed_fasta = fasta + (padding*(trimmed_thresh-seq_len))
        else:
            trimmed_fasta = fasta 
        trimmed_fastas.append(trimmed_fasta)    
    return trimmed_fastas


if __name__ == "__main__": 

    max_peptide_len = 15
    padding = '<pad>'

    model_saving_path = "/home_exp_2/jiani.ma/mhc/llm_pretrain/model_saving/model_dict_for_cpu.pkl"
    checkpoint = torch.load(model_saving_path)   #,map_location='cpu')

    finetuned_peptide_model = checkpoint.esm2_model

    peptide_df_path = "/home_exp_2/jiani.ma/mhc/pmhchat-pretrained/NetMHCIIpan-3.2-pretrained/peptide_df.xlsx"
    
    # read the peptide ids and seqs 
    peptide_df = pd.read_excel(peptide_df_path)        
    peptide_ids = peptide_df['Peptide'].tolist()  # will be in use later 
    seqs = peptide_df['Seq'].tolist()
    trimmed_seqs = trimming_fasta(seqs,max_peptide_len,padding)
    print("There exists {} peptide samples".format(len(peptide_ids))) 

    data = list(zip(peptide_ids,trimmed_seqs))  
    peptide_info_dict_dir = "/home_exp_2/jiani.ma/mhc/pmhchat-pretrained/pretrained_peptide_info_dict/"

    for i in range(len(data)):
        
        peptide_info_dict = {}
        item_form = []
        item = data[i] 
        item_form.append(item)      #[(),()]
        peptide = peptide_ids[i]

        finetuned_peptide_model.eval()
        representations,contact_map = finetuned_peptide_model(item_form)    
        representations = representations.detach().cpu()
        contact_map = contact_map.detach().cpu()

        # ！！！ check the size and shape of contact map
        peptide_info_dict['peptide'] = peptide
        peptide_info_dict['feature_representation'] = representations.squeeze(0)  
        peptide_info_dict['cmap'] = contact_map.squeeze(0)  
        file_name = peptide + ".pt"  
        file_path = os.path.join(peptide_info_dict_dir,file_name) 
        torch.save(peptide_info_dict,file_path) 
        print('Transformation for peptide {} is complete!'.format(peptide))
