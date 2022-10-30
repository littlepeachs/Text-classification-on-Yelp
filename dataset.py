import json
import os
import copy
import re
from torch.utils.data import Dataset, DataLoader
import torch
import sys

from traitlets import Float
from dictionary import Dictionary
import csv
import collections
import nltk
import logging
import random
import numpy as np
import csv
import string
from collections import defaultdict
# nltk.download("stopwords")

def collate_fn(batch):
    max_length = 0
    for token_id, label, length in batch:
        max_length = max(length, max_length)
    token_ids = torch.zeros(len(batch), max_length, dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.long)
    for i, (token_id, label, length) in enumerate(batch):
        token_ids[i] = torch.cat([token_id, torch.tensor([0] * (max_length - len(token_id)), dtype=torch.long)])
        labels[i] = label
    return token_ids, labels

class CLSDataset(Dataset):
    def __init__(self, data_path="./yelp_small/", dictionary=None, split='train', block_size=0):

        self.filename = os.path.join(data_path, "{}.csv".format(split))
        self.data = []
        self.dictionary = dictionary
        self.padding_idx = self.dictionary.pad()
        self.vocab_size = len(dictionary)
        self.block_size = block_size
        self.max_length = 0
        self.lens = []

        with open(self.filename) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                label = int(row[0])
                raw_text = row[1]
                tokens = row[1].replace('\\n', ' ')
                tokens = re.sub(r'[^a-zA-Z\s]', ' ', string=tokens)
                tokens = tokens.strip().split(' ')
                
                for i in reversed(range(len(tokens))):
                    if tokens[i] == '': 
                        assert tokens.pop(i) == ''
                        continue

                    tokens[i] = tokens[i].lower()
                    if tokens[i] not in self.dictionary.indices.keys():
                        tokens[i] = '<unk>'
                token_id = self.dictionary.encode_line(tokens)
                self.max_length = max(self.max_length, len(tokens))
                self.lens.append(len(tokens))
                self.data.append((raw_text, tokens, token_id, label, len(token_id)))
        self.data = sorted(self.data, key = lambda x:x[4])
        self.idx = list(range(len(self.data)))
        if self.block_size > 0:
            l = 0
            while (l < len(self.idx)):
                r = min(len(self.idx), l + self.block_size)
                tmp = self.idx[l:r]
                random.shuffle(tmp)
                self.idx[l:r] = tmp
                l = r
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''return a batch of samples'''
        raw_text, tokens, token_id, label, length = self.data[self.idx[index]] 
        return token_id, label, length


class BOWDataset(Dataset):
    def __init__(self, data_path="./yelp_small/", dictionary=None, split='train', n=0):
        # n for n_gram
        self.filename = os.path.join(data_path, "{}.csv".format(split))
        self.data = []
        self.label = []
        self.dictionary = dictionary
        self.vocab_size = len(dictionary) 
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.preprocess()
        self.transform2tensor()
        self.data = torch.FloatTensor(self.data)
        self.label=torch.tensor(self.label)
          
    def preprocess(self):
        punctuation_string = string.punctuation
        with open(self.filename,'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.label.append(int(row[0]))
                self.data.append(row[1])
        for i in range(len(self.data)):
            sentence = self.data[i]
            sen_split = sentence.split()
            j=0
            while j < len(sen_split):
                contain_num= 0
                k=0
                while k < len(sen_split[j]):
                    if sen_split[j][k] in punctuation_string:
                        sen_split[j] = sen_split[j][:k]+sen_split[j][k+1:]
                        continue
                    if sen_split[j][k] in ['0','1','2','3','4','5','6','7','8','9']:
                        contain_num=1
                        break
                    k+=1
                sen_split[j] = sen_split[j].lower()
                if (sen_split[j] in self.stopwords or contain_num==1):
                    sen_split.pop(j)
                    continue
                j+=1
            sentence = " ".join(sen_split)
            self.data[i] = sen_split  
        
    def transform2tensor(self):
        for i in range(len(self.data)):
            sen_tensor = np.zeros(self.vocab_size)
            words = self.data[i]
            for word in words:
                if word in self.dictionary.indices:
                    sen_tensor[self.dictionary.indices[word]]+=1
                else:
                    sen_tensor[self.dictionary.indices['<unk>']]+=1
            self.data[i]=sen_tensor


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # bow_freq_feature and label
        return [self.data[index],self.label[index]]


class FastTextDataset(Dataset):
    def __init__(self, data_path="./yelp_small/", dictionary=None, split='train', n=2,n_gram_size=None):

        self.filename = os.path.join(data_path, "{}.csv".format(split))
        self.data = []
        self.label = []
        # self.max_length = 200 
        self.n_gram = n
        self.dictionary = dictionary
        self.hash_size = 1000     
        self.padding_idx = self.dictionary.pad()
        self.vocab_size = len(dictionary)
        self.total_size = self.hash_size+self.vocab_size
        self.lens = []  # record the ith sentence length
        self.preprocess()
        self.transform2tensor()
        print("success")

    def hash32(self,hash_str):
        hashres = 2166136261
        FNV_prime = 16777619
        for char in hash_str:
            hashres = hashres ^ ord(char)
            hashres = hashres * FNV_prime

        return hashres & ((1 << 32) - 1)

    #Lazy mod mapping method:
    def hashLazy32(self,hash_str,range):
        return self.hash32(hash_str) % range

    def n_gram_feature(self,x):
        init_length = len(x)
        for i in range(init_length-1):
            n_gram = x[i]+" "+x[i+1]
            x.append(n_gram)
        return x

    def preprocess(self):
        with open(self.filename,'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.label.append(int(row[0]))
                self.data.append(row[1])
        for i in range(len(self.data)):
            words = self.data[i].split()
            self.lens.append(len(words))
            # if len(words)>=self.max_length:
            #     words = words[:self.max_length]
            # while len(words)<self.max_length:
            #     words.append('<pad>')
            words = self.n_gram_feature(words)
            self.data[i] = words
    
    def transform2tensor(self):
        for i in range(len(self.data)):
            for j in range(self.lens[i]):
                self.data[i][j] = self.data[i][j].lower()
                if self.data[i][j] in self.dictionary.indices:
                    self.data[i][j] = self.dictionary.indices[self.data[i][j]]
                else:
                    self.data[i][j] = self.dictionary.indices["<unk>"]

            for j in range(self.lens[i],len(self.data[i])):
                self.data[i][j]=self.hashLazy32(self.data[i][j],self.hash_size)+self.vocab_size
            self.data[i] = torch.FloatTensor(self.data[i])
            self.lens[i] = len(self.data[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index],self.label[index],self.lens[index]


    
    
    

        

        
        
