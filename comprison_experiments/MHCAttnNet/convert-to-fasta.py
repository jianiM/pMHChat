# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 18:24:13 2024
@author: amber
"""

import pandas as pd


def convert_mhc_to_fasta(input_excel, output_fasta):
    # Load the Excel file
    df = pd.read_excel(input_excel)
    # Open the output file for writing in FASTA format
    with open(output_fasta, "w") as f:
        for _, row in df.iterrows():
            allele = row['mhc-allele']
            pseudosequence = row['pseudosequence']
            # Write each sequence in FASTA format
            f.write(f">{allele}\n")
            f.write(f"{pseudosequence}\n")


def convert_peptide_to_fasta(input_excel, output_fasta):
    # Load the Excel file
    df = pd.read_excel(input_excel)
    # Open the output file for writing in FASTA format
    with open(output_fasta, "w") as f:
        for _, row in df.iterrows():
            peptide = row['Peptide']
            seq = row['Seq']
            # Write each sequence in FASTA format
            f.write(f">{peptide}\n")
            f.write(f"{seq}\n")


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

    # MHC
    mhc_path = "/home_exp_2/jiani.ma/mhc/NetMHCIIpan-3.2/mhc_df.xlsx" 
    mhc_output_fasta = "/home_exp_2/jiani.ma/mhc/comparison-experiments/MHCAttenNet/dataset/mhc_corpus.fasta"
    convert_mhc_to_fasta(mhc_path, mhc_output_fasta)
    
    # padding= 'X'
    # max_peptide_len = 15 
    # orig_peptide_path = "/home_exp_2/jiani.ma/mhc/NetMHCIIpan-3.2/peptide_df.xlsx"
    # peptide_output_fasta = '/home_exp_2/jiani.ma/mhc/comparison-experiments/MHCAttenNet/dataset/peptide_corpus.fasta'

    # # Convert the Excel data to FASTA format
    # peptide_df = pd.read_excel(orig_peptide_path)
    # peptide_ids = peptide_df['Peptide'].tolist()  # will be in use later 
    # seqs = peptide_df['Seq'].tolist()
    # trimmed_seqs = trimming_fasta(seqs,max_peptide_len,padding)

    # peptide_df_trimmed = pd.DataFrame({'Peptide': peptide_ids,'Seq': trimmed_seqs})
    
    # peptide_path = "/home_exp_2/jiani.ma/mhc/comparison-experiments/MHCAttenNet/dataset/trimmed_peptides.xlsx"
    # peptide_df_trimmed.to_excel(peptide_path)
    # convert_peptide_to_fasta(peptide_path, peptide_output_fasta)