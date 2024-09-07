
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
        sequence = self.seq[idx]
        seq = [self.seq_map[s] for s in sequence]
        seq = torch.Tensor(seq).to(torch.int32)
        
        output = dict()
        output['input_ids'] = seq
        sequence_mask = []
        for i in range(len(seq)) :
            mask = torch.ones(len(seq))
            mask[max(i-1, 0)] = 0
            mask[min(i+1, len(seq)-1)] = 0
            sequence_mask.append([self.reactivity_map[sequence[i]][s] for s in sequence])
            torch.Tensor(sequence_mask[i])*mask
        
        output['length'] = len(sequence)
        output['decoder_mask'] = torch.tensor(sequence_mask)
        output['labels'] = torch.stack([self.react_DMS[idx], self.react_2A3[idx]],-1)
        return output