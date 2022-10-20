import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_size=512, hidden_size=512, padding_idx=None):
        super(CNN, self).__init__()
        #########################################  Your Code  ###########################################
        # todo
        # implement text cnn
        # the output shape should be batch * 5

        raise NotImplementedError
        #################################################################################################

    def forward(self, inputs):
        #inputs  Batch * seq_length

        #########################################  Your Code  ###########################################
        # todo
        # implement text cnn
        # the output logits shape should be batch * 5

        raise NotImplementedError
        #################################################################################################
        return outputs





