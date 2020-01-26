import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import datetime as dt
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from modules import encoder, decoder

class da_rnn(nn.Module):
    def __init__(self, train_data, encoder_hidden_size, decoder_hidden_size, T, learning_rate, batch_size)
        super(da_rnn, self).__init__()
        input_size = train_data.shape[1]

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.T = T

        self.encoder = encoder(input_size, encoder_hidden_size, T)
        self.decoder = decoder(encoder_hidden_size, decoder_hidden_size, T)

        encoder_optimizer = optim.Adam(
        params=[p for p in encoder.parameters() if p.requires_grad],
        lr=learning_rate)
        decoder_optimizer = optim.Adam(
        params=[p for p in decoder.parameters() if p.requires_grad],
        lr=learning_rate)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)