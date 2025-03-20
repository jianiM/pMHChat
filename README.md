![workflow]https://github.com/jianiM/pMHChat/blob/main/workflow.png


# Requirements
+ python == 3.9.19 
+ torch == 2.3.1+cu118
+ torch-geometric == 2.6.1
+ biopython == 1.84
+ fair-esm == 2.0.0
+ numpy == 1.26.4 
+ pandas == 1.5.3
+ openpyxl == 3.1.5
+ scikit-learn == 1.5.0 
+ scipy == 1.13.1
+ networkx == 3.0
+ matplotlib == 3.8.4 
+ seaborn == 0.13.2


# Datasets

+ BD2016: 5-fold CV dataset(https://services.healthtech.dtu.dk/suppl/immunology/NetMHCIIpan-3.2/)
+ BD2024: Processed independent test set(http://tools.iedb.org/auto_bench/mhcii/weekly/)
+ BC2015: Crystal Structures of 51 pMHC complex


# Step-by-step training for pMHChat 
1. Pretrain and Fine-tune
    + python ./pretrained/main.py 
    
    
2. Generate MHC position embeddings, peptide residue embeddings and contact map 
    + python create_mhc_features.py 
    + python create_peptide_features.py 

3. Train and Test models in the 5-fold CV scheme 
    + python main.py 












