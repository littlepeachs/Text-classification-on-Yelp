import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size=512, hidden_size=512, num_layers=1, padding_idx=None):
        super(LSTM, self).__init__()
        #########################################  Your Code  ###########################################
        # todo
        # implement lstm
        # the output shape should be batch * 5

        raise NotImplementedError
        #################################################################################################

    def forward(self, inputs, last_hidden=None):
        #inputs  Batch * seq_length

        #########################################  Your Code  ###########################################
        # todo
        # implement lstm
        # the output logits shape should be batch * 5

        raise NotImplementedError
        #################################################################################################
        return outputs





