import torch
import numpy as np
import pandas as pd
from torch import Tensor, nn
from tqdm.notebook import tqdm
from torchaudio.models import Conformer
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class encoder_model(nn.Module) :
    def __init__(self, kernel_size: int, num_channels: int, num_layers: int, feed_forward: int, num_heads: int) :
        super().__init__()

        self.embedding = nn.Sequential(nn.Embedding(457, num_channels//4), nn.ReLU(), nn.Linear(num_channels//4, num_channels//2), 
                                        nn.ReLU(), nn.Linear(num_channels//2, num_channels))

        self.encoder =  Conformer(num_channels, num_heads, feed_forward, num_layers, kernel_size)     

    def forward(self, input_ids, length, mask) :

        embedding = self.embedding(input_ids)*mask
        encoded, _ = self.encoder(embedding, length)

        return encoded
    
