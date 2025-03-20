# pMHChat, Characterizing the Interactions Between MHC Class II Molecules and Peptides with LLMs and Deep Hypergraph Learning 
+ This repository contains the source code for the paper pMHChat, Characterizing the Interactions Between MHC Class II Molecules and Peptides with LLMs and Deep Hypergraph Learning. 

![workflow](https://github.com/jianiM/pMHChat/blob/main/workflow.png)

pMHChat is developed for MHC II-peptide binding prediction with pLMs and hyperconv. 

# Datasets and model checkpoints
+ BD2016: 5-fold CV dataset(https://services.healthtech.dtu.dk/suppl/immunology/NetMHCIIpan-3.2/)
+ BD2024: Processed independent test set(http://tools.iedb.org/auto_bench/mhcii/weekly/)
+ BC2015: Crystal Structures of 51 pMHC complex
+ checkpoint for fine-tuning stage can be found at https://zenodo.org/records/15057065

# Usage
## 1. Create the environment by conda
+ conda create -n py310 python=3.10
+ conda activate py310
+ pip3 install torch torchvision torchaudio  #for cuda 12.4
+ conda install pyg -c pyg
+ pip install -r requirements.txt 

## 2. pLMs Fine-tuning procedure
+ # Driven by binding reactivity prediction task, pretrain and fine-tune ESM-MSA-1B and ESM-2.
+ python llm_pretrain/main.py  

## 3. Residue Feature Generation 
+ # Generate the residue-level feature for MHC pseudosequence and peptide sequence, as well as residue contact map of peptide
+ create_mhc_features.py
+ create_peptide_features.py
    
## 4. Train or test the models wth 5-fold CV scheme
+ python main.py 











