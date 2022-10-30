import torch
from torch import nn
import torch.nn.functional as F 

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, 
                 padding_idx=None):
        
        super().__init__()

        self.n_filters =20

        self.output_dim = 5

        self.filter_sizes = [2,3,4]
                
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = padding_idx)
        
        # self.conv2d = nn.Conv2d(in_channels = 1,out_channels = self.n_filters,kernel_size = (2, embedding_dim))

        self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1,out_channels = self.n_filters,kernel_size = (fs, embedding_dim)) for fs in self.filter_sizes])
        
        self.fc = nn.Linear(len(self.filter_sizes)*self.n_filters, self.output_dim)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, text):
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
        
        embedded = self.dropout(embedded.unsqueeze(1))
        
        conved = [torch.sigmoid(conv(embedded)).squeeze(3) for conv in self.convs]
                
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))

        output = self.fc(cat)           
        return output