from itertools import count
import os
import numpy as np
import pandas as pd
from skimage import transform

import torch
from torch.utils.data import Dataset

# # local functions
# from dataset.utils import *


class DepressionDataset(Dataset):

    def __init__(self,root_dir,mode,transform,SMP=False):
        super(DepressionDataset, self).__init__()
        self.mode = mode
        self.root_dir = root_dir
        self.transform = transform
        self.smp = SMP
        self.IDs = np.load(os.path.join(self.root_dir, 'ID_gt.npy'),allow_pickle=True)
        self.phq_binary_gt = np.load(os.path.join(self.root_dir, 'phq_binary_gt.npy'),allow_pickle=True)
        self.num = np.load(os.path.join(self.root_dir, 'phq_no_gt.npy'),allow_pickle=True)
        
          


    def __len__(self):
        return len(self.IDs)

    def __iter__(self):
        return iter(self.IDs)

    def __getitem__(self, idx):
    
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = os.path.join(self.root_dir,'mel-spectrogram')
        audio_file = np.sort(os.listdir(audio_path))[idx]
        audio_file = str(audio_file)
        audio = np.load(os.path.join(audio_path, audio_file),allow_pickle=True)

        session = {'ID': self.IDs[idx].astype(float),
                   'phq_binary_gt': self.phq_binary_gt[idx],
                   'num': self.num[idx],
                   'audio': audio}
        

        if self.transform:
            session = self.transform(session)

        return session

class ToTensor(object):
    """Convert ndarrays in sample to Tensors or np.int to torch.tensor."""

    def __init__(self, mode):
        # assert mode in ["train", "validation", "test"], \
        #     "Argument --mode could only be ['train', 'validation', 'test']"
        
        self.mode = mode

    def __call__(self, session):
        if self.mode == 'train' or self.mode == 'test':
            converted_session = {'ID': session['ID'],
                                 'phq_binary_gt': torch.tensor(session['phq_binary_gt'],dtype=int),
                                 'num': torch.tensor(session['num'],dtype=int),
                                 'audio': torch.from_numpy(session['audio']).type(torch.FloatTensor)}
        else:
            converted_session = {'audio': torch.from_numpy(session['audio']).type(torch.FloatTensor)}
        
        
       
        return converted_session


class SMPDataset(Dataset):

    def __init__(self, root_dir, mode, transform):
        super(SMPDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.IDs = os.listdir(os.path.join(root_dir,mode))

    def __len__(self):
        return len(self.IDs)

    def __iter__(self):
        return iter(self.IDs)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = os.path.join(self.root_dir, self.mode)
        audio_file = np.sort(os.listdir(audio_path))[idx]
        audio_file = str(audio_file)
        audio = np.load(os.path.join(audio_path, audio_file), allow_pickle=True)

        session = {'audio': audio}

        if self.transform:
            session = self.transform(session)

        return session


