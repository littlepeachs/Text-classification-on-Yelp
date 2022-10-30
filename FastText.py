import torch
import torch.nn as nn
import torch.nn.functional as F 

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_size=512, padding_idx=None):
        super(FastText, self).__init__()
        self.embed = nn.Embedding(vocab_size,embedding_size,padding_idx)
        self.fc = nn.Linear(embedding_size,5)
        self.dropout = nn.Dropout(0.6)

    def forward(self, inputs):
        embed = self.embed(inputs)
        embed = self.dropout(embed)
        pooled = F.avg_pool2d(embed, (embed.shape[1], 1)).squeeze(1) 
        outputs = self.fc(pooled)
        return outputs





