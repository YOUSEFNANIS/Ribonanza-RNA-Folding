import torch
import numpy as np
import pandas as pd
from torch import Tensor, nn
from tqdm.notebook import tqdm
from torchaudio.models import Conformer
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class RNA_model(nn.Module) :
    def __init__(self, kernel_size: int, num_channels: int, num_layers: int, feed_forward = 1024, num_heads = 16) :
        super().__init__()

        self.postional_embedding = nn.Sequential(nn.Embedding(457, num_channels//4), nn.Sigmoid(), nn.Linear(num_channels//4, num_channels//2), nn.ReLU())        
        self.base_embedding = nn.Sequential(nn.Embedding(5, num_channels//4), nn.Sigmoid(), nn.Linear(num_channels//4, num_channels//2), nn.ReLU())
        
        self.feed_forward = nn.Sequential(nn.Linear(num_channels, num_channels*2), nn.Sigmoid(), nn.Linear(num_channels*2, num_channels), nn.ReLU())
        
        self.encoder =  Conformer(num_channels, num_heads, feed_forward, num_layers, kernel_size)     
        self.result = nn.Sequential(nn.Sigmoid(), nn.Linear(num_channels, num_channels//2), nn.Linear(num_channels//2, num_channels//4), 
                                    nn.ReLU(), nn.Linear(num_channels//4, num_channels//8), nn.ReLU(), nn.Linear(num_channels//8, 2), nn.ReLU())
        self.loss = nn.L1Loss()
        
    def forward(self, input_ids, length, mask, labels=None) :

        mask = torch.unsqueeze(mask, dim=-1)
        max_len = torch.max(length)
        mask = mask[:, :max_len]
        input_ids = input_ids[:, :max_len]
        
        positional_embedding = self.postional_embedding(input_ids)*mask
        base_embedding = self.base_embedding(input_ids)*mask
        embedding = torch.concat((positional_embedding, base_embedding), -1)
        feed_forward = self.feed_forward(embedding) + embedding
        
        encoded, _ = self.encoder(feed_forward, length)
        output = self.result(encoded)*mask
        
        if labels is not None :

            y = labels[:, :max_len]
            cover = y != 0
            output *= cover
            loss = torch.unsqueeze( self.loss(output, y), dim=0)
            return loss
        
        return output
    
