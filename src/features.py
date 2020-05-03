import numpy as np
import glob
import os
import torch
import warnings

import torch.multiprocessing
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split

# this is really important - without this the program fails with a 
# "too many files open" error, at least on UNIX systems
torch.multiprocessing.set_sharing_strategy('file_system')

warnings.filterwarnings("ignore")

from multiprocessing import cpu_count
import fma_utils as utils



np.random.seed(0)

# path to all data and metadata
BASE_PATH = '../data/'

# @TODO Refactor class:
#           - naming is misleading, dataset is configured in two places
#           - move into the other dataset class and parameterize there
#           - (just moved global constants and feature extraction
#              stuff to here to parameterize instead of having to
#              reload the training notebook's kernel every time we
#              switch datasets)
class DatasetSettings:
    def __init__(self, data_name, metadata_name):
        # set paths to data and metadata
        self.data_dir = os.path.join(BASE_PATH, data_name)
        self.metadata_dir = os.path.join(BASE_PATH, metadata_name)
        # path to extracted spectograms and targets  (stored in npz)
        self.target_dir = os.path.join(self.data_dir, 'targets')
        # set env var for FMA utils.py script to know where to look
        # for audio data
        os.environ['AUDIO_DIR'] = self.data_dir
        # grab track and genre metadata
        self.tracks = utils.load('../data/fma_metadata/tracks.csv') #load tracks df
        self.genres = utils.load('../data/fma_metadata/genres.csv') #load genres df
        # grab paths to target files and a list of their track IDs
        self.target_paths = [*glob.iglob(os.path.join(self.target_dir,
            '*_targets.npz'), recursive=True)]
        self.tids = list(map(lambda x: int(os.path.splitext(
            os.path.basename(x).replace('_targets', ''))[0]),
            self.target_paths))
        # grab metadata for tracks available in the current subset of
        # data (metadata contains all tracks, data contains a subset
        # corresponding to the particular size, e.g. fma_small)
        self.tracks_subset = self.tracks['track'].loc[self.tids] #track name?
        self.genres_subset = self.tracks_subset['genre_top'] #genre for each track 
        self.artists_subset = self.tracks['artist'].loc[self.tids]
        # count genre occurences
        genre_counts = self.genres_subset.value_counts()
        self.genre_counts = genre_counts[genre_counts > 0]
        self.num_genres = len(self.genre_counts)
        # create genre lookups
        self.coded_genres = {genre: k for k, genre in enumerate(self.genre_counts.index)}
        self.coded_genres_reverse = {k: genre for genre, k in self.coded_genres.items()}
        # some fixed settings
        # X frames with 50% overlap = 2X-1 frames
        self.num_frames = 4
        self.total_frames = 2 * self.num_frames - 1
        self.frame_size = 1290 // self.num_frames

class FeatureDataset(Dataset):
    def __init__(self, settings: DatasetSettings, agfs=[], genre=True):
        super().__init__()
        self.agfs = agfs
        self.genre = genre
        self.settings = settings
        
    def __len__(self):
        return len(self.settings.target_paths)
    
    def __getitem__(self, idx):
        path = self.settings.target_paths[idx]
        tid = self.settings.tids[idx]
        
        # argmax these
        features = {}
        
        with np.load(path) as data:
            for agf in self.agfs:
                lda_vec = data[agf]
                features[agf] = lda_vec.argmax() #torch.from_numpy((lda_vec == lda_vec.max())*1) #argmax the output of Latent Dirichlet Allocation
            
            mel = data['mel']
        
        if self.genre:
            features['genre'] = self.settings.coded_genres[self.settings.tracks_subset['genre_top'][tid]]
        
        return mel, features

class FramedFeatureDataset(FeatureDataset):
    def __len__(self):
        return self.settings.total_frames * super().__len__()
    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        
        song_idx, frame = divmod(idx, self.settings.total_frames)
        mel, features = super().__getitem__(song_idx)
        
        shift, half_shift = divmod(frame, 2)
        i = shift * self.settings.frame_size + half_shift * self.settings.frame_size // 2
        
        # add channel dimension so its 1x128x(frame_size)
        mel_frame = np.expand_dims(mel[:, i:i + self.settings.frame_size], axis=0)
        
        return mel_frame, features

def get_data_loaders(dataset, batch_size, valid_split):
    dataset_len = len(dataset)
    
    # split dataset
    valid_len = int(dataset_len * valid_split)
    train_len = dataset_len - valid_len
    
    train_dataset, valid_dataset = random_split(dataset, [train_len, valid_len])
    
    # disable if it fucks things up but if it doesnt its apparently rly good 
    pin_memory = True
    
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=cpu_count(),
                              # sampler=train_sampler,
                              shuffle=True,
                              pin_memory=pin_memory)
    
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              num_workers=cpu_count(),
                              shuffle=False,
                              pin_memory=pin_memory)
    
    return train_loader, valid_loader

if __name__ == '__main__':
    
    dataset = FramedFeatureDataset(['subgenres'], False)
    print(len(dataset))
    
    train_loader, valid_loader = get_data_loaders(dataset, 64, 0.15)
    print(len(train_loader))
    
    for _ in range(2):
        for i, batch in enumerate(train_loader):
            if i % 30 == 0:
                print(i, '/', len(train_loader))
