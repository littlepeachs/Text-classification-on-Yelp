import torch.nn.functional as F 
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size=512, hidden_size=512, num_layers=1, padding_idx=None):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size,embedding_size,padding_idx)
        self.bilstm = nn.LSTM(embedding_size,hidden_size,num_layers,dropout=0.6,batch_first=True,bidirectional=True)
        self.FC = nn.Linear(2*hidden_size,5)
        self.dropout = nn.Dropout(0.6)

    def forward(self, inputs, last_hidden=None):
        embed = self.embed(inputs)
        embed = self.dropout(embed)
        outputs,(h,c) = self.bilstm(embed)
        outputs = self.dropout(outputs)
        outputs = F.avg_pool2d(outputs, (embed.shape[1], 1)).squeeze(1) 
        
        outputs = self.FC(outputs)
        return outputs





