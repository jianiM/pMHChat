import sys, os, json
import argparse
import pandas as pd
import numpy as np
from gensim.models import word2vec
from Bio import SeqIO
from gensim.models.keyedvectors import KeyedVectors
from tqdm import tqdm
import torchtext
torchtext.disable_torchtext_deprecation_warning()
import torch 
import torch.nn as nn 



def split_ngrams(seq, n):
    """
    Split sequence into non-overlapping n-grams
    'ATSKLGH' --> [['ATS','KLG'],['TSK','LGH'],['SKL']]
    """
    kmers = list()
    for i in range(n):
        kmers.append(zip(*[iter(seq[i:])]*n))
    str_ngrams = list()
    for ngrams in kmers:
        x = list()
        for ngram in ngrams:
            x.append("".join(ngram))
        str_ngrams.append(x)
    return str_ngrams


def generate_corpusfile(corpus_fname, n, out):
    with open(out, "w") as f:
        for r in tqdm(SeqIO.parse(corpus_fname, "fasta")):
            if("B" not in r.seq and "J" not in r.seq and "O" not in r.seq and "U" not in r.seq and "Z" not in r.seq):    # sanity check to remove invalid amino acids
                ngram_patterns = split_ngrams(r.seq, n)
                for ngram_pattern in ngram_patterns:
                    f.write(" ".join(ngram_pattern) + "\n")    # Take all the sequences and split them into kmers


class MHCProtVec(word2vec.Word2Vec):
    def __init__(self, corpus_fname=None, n=1, vector_size=100, out="/home_exp_2/jiani.ma/mhc/comparison-experiments/MHCAttenNet/dataset/mhc_output_corpus.txt", sg=1, window=10, min_count=1, workers=9):
        self.n = n
        self.vector_size = vector_size
        self.corpus_fname = corpus_fname
        self.sg = sg
        self.window = window
        self.workers = workers
        self.out = out
        self.vocab = min_count

        if(corpus_fname is not None):
            if(not os.path.isfile(out)):
                print("-- Generating corpus --")
                generate_corpusfile(corpus_fname, n, out)
            else:
                print("-- Corpus File Found --")
        
        self.corpus = word2vec.Text8Corpus(out)
        print("-- Corpus Setup Successful --")

    def word2vec_init(self, vectors_txt, model_weights):
        print("-- Initializing Word2Vec model --")
        print("-- Training the model --")
        self.m = word2vec.Word2Vec(self.corpus, vector_size=self.vector_size, sg=self.sg, window=self.window, min_count=self.vocab, workers=self.workers)
        self.m.wv.save_word2vec_format(vectors_txt)
        self.m.save(model_weights)
        print("-- Saving Model Weights to : %s " % (vectors_txt))

    def load_protvec(self, model_weights):
        print("-- Load Word2Vec model --")
        self.m = word2vec.Word2Vec.load(model_weights)
        return self.m


class PeptideProtVec(word2vec.Word2Vec):
    def __init__(self, corpus_fname=None, n=3, vector_size=100, out="/home_exp_2/jiani.ma/mhc/comparison-experiments/MHCAttenNet/dataset/peptide_output_corpus.txt", sg=1, window=5, min_count=1, workers=9):
        self.n = n
        self.vector_size = vector_size
        self.corpus_fname = corpus_fname
        self.sg = sg
        self.window = window
        self.workers = workers
        self.out = out
        self.vocab = min_count

        if(corpus_fname is not None):
            if(not os.path.isfile(out)):
                print("-- Generating corpus --")
                generate_corpusfile(corpus_fname, n, out)
            else:
                print("-- Corpus File Found --")
        
        self.corpus = word2vec.Text8Corpus(out)
        print("-- Corpus Setup Successful --")

    def word2vec_init(self, vectors_txt, model_weights):
        print("-- Initializing Word2Vec model --")
        print("-- Training the model --")
        self.m = word2vec.Word2Vec(self.corpus, vector_size=self.vector_size, sg=self.sg, window=self.window, min_count=self.vocab, workers=self.workers)
        self.m.wv.save_word2vec_format(vectors_txt)
        self.m.save(model_weights)
        print("-- Saving Model Weights to : %s " % (vectors_txt))

    def load_protvec(self, model_weights):
        print("-- Load Word2Vec model --")
        self.m = word2vec.Word2Vec.load(model_weights)
        return self.m


if __name__ == "__main__":

    # 1. FOR MHC 
    
    # corpus_fname = "/home_exp_2/jiani.ma/mhc/comparison-experiments/MHCAttenNet/dataset/mhc_corpus.fasta"
    # vectors = '/home_exp_2/jiani.ma/mhc/comparison-experiments/MHCAttenNet/dataset/1-gram-vectors.txt' 
    # model_weights ='/home_exp_2/jiani.ma/mhc/comparison-experiments/MHCAttenNet/dataset/1-gram-model-weights.mdl'
    
    # # firstly running these two lines:  
    # mhc_model = MHCProtVec(corpus_fname=corpus_fname)
    # mhc_model.word2vec_init(vectors_txt=vectors, model_weights=model_weights)
    
    # Then
    # mhc_model = MHCProtVec()
    # mhc_model.load_protvec(model_weights=model_weights)
    
    
    # #2. FOR Peptide 
    # peptide_corpus_fname = '/home_exp_2/jiani.ma/mhc/comparison-experiments/MHCAttenNet/dataset/peptide_corpus.fasta'
    # peptide_vectors = '/home_exp_2/jiani.ma/mhc/comparison-experiments/MHCAttenNet/dataset/3-gram-vectors.txt' 
    # peptide_model_weights ='/home_exp_2/jiani.ma/mhc/comparison-experiments/MHCAttenNet/dataset/3-gram-model-weights.mdl' 
    
    # # firstly running these two lines:  
    # model = PeptideProtVec(corpus_fname=peptide_corpus_fname)
    # model.word2vec_init(vectors_txt=peptide_vectors, model_weights=peptide_model_weights)
    
    # model = PeptideProtVec()
    # model.load_protvec(model_weights=peptide_model_weights)
    
    # protvec = PeptideProtVec()
    # weight = protvec.load_protvec(model_weights='/home_exp_2/jiani.ma/mhc/comparison-experiments/MHCAttenNet/dataset/3-gram-model-weights.mdl')

    # peptide_vocab = protvec.m.wv.key_to_index  # Adjust based on gensim version
    # print('peptide_vocab:',peptide_vocab)
    # seq = "QRAQATMLAETYFGV" 
    
    # print('peptide_vocab:',peptide_vocab)
    # peptide_embeddings = torch.tensor([protvec.m.wv[word] for word in peptide_vocab], dtype=torch.float)
    # #print('peptide_embeddings:',peptide_embeddings.size())


    protvec = PeptideProtVec()  
    pep_weight = protvec.load_protvec(model_weights='/home_exp_2/jiani.ma/mhc/comparison-experiments/MHCAttenNet/dataset/3-gram-model-weights.mdl')  
    # achieve pretrained word2vec embeddings 
    pep_embed = torch.tensor(protvec.m.wv.vectors, dtype=torch.float32)  
    peptide_embedding_layer = nn.Embedding.from_pretrained(pep_embed, freeze=False)  

    print('peptide_embedding_layer:',peptide_embedding_layer)
    # pep_embeddings = peptide_embedding_layer(torch.tensor(pep_indices, dtype=torch.long))  
    


   



