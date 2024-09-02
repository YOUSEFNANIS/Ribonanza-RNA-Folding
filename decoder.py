import torch
import numpy as np
import pandas as pd
from . import encoder
from typing import Any
from torch import Tensor, nn
from tqdm.notebook import tqdm
from dataclasses import dataclass
from torch.utils.data import Dataset
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

data_encoder = encoder.encoder_model(kernel_size = 7, num_channels=256, num_layers=10, num_heads=16)

class RNA_Dataset(Dataset):
    def __init__(self, df, size=206):
        self.seq_map = {'A':1,'C':2,'G':3,'U':4}
        self.size = size
        self.reactivity_map = {'A' : {'A' : 0, 'C' : 0, 'G' : 0, 'U' : 2}, 'C' : {'A' : 0, 'C' : 0, 'G' : 3, 'U' : 0}, 0 : {'A' : 0, 'C' : 0, 'G' : 0, 'U' : 0},
                        'U' : {'A' : 2, 'C' : 0, 'G' : 1, 'U' : 0}, 'G' : {'A' : 0, 'C' : 3, 'G' : 0, 'U' : 1}}
        
        df_2A3 = df.loc[df.experiment_type=='2A3_MaP'].reset_index(drop=True)
        df_DMS = df.loc[df.experiment_type=='DMS_MaP'].reset_index(drop=True)
        
        self.seq = df_2A3['sequence'].values
        self.react_2A3 = torch.from_numpy(df_2A3[[c for c in df_2A3.columns if 'reactivity_0' in c]].values)
        self.react_DMS = torch.from_numpy(df_DMS[[c for c in df_DMS.columns if 'reactivity_0' in c]].values)

        df = None
        df_2A3 = None
        df_DMS = None

    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, idx):
        seq = self.seq[idx]
        length = len(seq)
        seq = [self.seq_map[s] for s in seq]
        seq = torch.Tensor(seq).to(torch.int32)
        mask = torch.zeros(self.size, dtype=torch.bool)
        #mask[:len(seq)] = 1
        
        output = dict()
        output['input_ids'] = seq.unsqueeze(0).repeat(seq.shape[0], 1, 1)
        output['length'] = length
        sequence_mask = []
        for i in output['input_ids'] :
            sequence_mask.append([self.reactivity_map[s] for s in seq])

        output['decoder_mask'] = torch.tensor(sequence_mask)
        output['mask'] = mask
        output['labels'] = torch.stack([self.react_DMS[idx], self.react_2A3[idx]],-1)
        
        return output
    
"""@dataclass
class decoder_collator:
    processor: Any
    reactivity_map = {'A' : {'A' : 0, 'C' : 0, 'G' : 0, 'U' : 2}, 'C' : {'A' : 0, 'C' : 0, 'G' : 3, 'U' : 0}, 0 : {'A' : 0, 'C' : 0, 'G' : 0, 'U' : 0},
                        'U' : {'A' : 2, 'C' : 0, 'G' : 1, 'U' : 0}, 'G' : {'A' : 0, 'C' : 3, 'G' : 0, 'U' : 1}}
        
    def __call__(self, features):
        length = features['length']
        sequence = features['sequence']
        mask = features['mask']
        mask = torch.unsqueeze(mask, dim=-1)
        max_len = torch.max(length)
        mask = mask[:, :max_len]
        input_ids = sequence[:, :max_len]

        with torch.eval() :
            data = encoder(input_ids, length, mask)

        batch = {}
        sequence_mask = []
        for i in features['input_ids'] :
            sequence_mask.append([self.reactivity_map[s] for s in sequence])

        batch['decoder_mask'] = torch.tensor(sequence_mask)
        batch['memory'] = data.unsqueeze(0).repeat(data.shape[0], 1, 1)
        batch['tgt'] = sequence.unsqueeze(1)
        batch['labels'] = features['labels']
        batch['base_mask'] = features['base_mask']
        return batch"""