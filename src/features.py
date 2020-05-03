import glob
import os
import torch
import warnings

import torch.multiprocessing
import torch.nn.functional as F

# this is really important - without this the program fails with a 
# "too many files open" error, at least on UNIX systems
torch.multiprocessing.set_sharing_strategy('file_system')

warnings.filterwarnings("ignore")

import numpy as np

from multiprocessing import cpu_count
import fma_utils as utils

np.random.seed(0)

audio_dir = '../data/fma_medium' #path to audio files
target_dir = os.path.join(audio_dir, 'targets') #path to extracted spectograms and targets  (stored in npz)

os.environ['AUDIO_DIR'] = audio_dir #idk what this does

tracks = utils.load('../data/fma_metadata/tracks.csv') #load tracks df
genres = utils.load('../data/fma_metadata/genres.csv') #load genres df

target_paths = [*glob.iglob(os.path.join(target_dir, '*_targets.npz'), recursive=True)] #list paths to targets
tids = list(map(lambda x: int(os.path.splitext(os.path.basename(x).replace('_targets', ''))[0]), target_paths)) #list of track ids

tracks_subset = tracks['track'].loc[tids] #track name?
genres_subset = tracks_subset['genre_top'] #genre for each track 
artists_subset = tracks['artist'].loc[tids]

from torch.utils.data import Dataset, DataLoader, random_split

genre_counts = genres_subset.value_counts()
genre_counts = genre_counts[genre_counts > 0]

print(genre_counts)

coded_genres = {genre: k for k, genre in enumerate(genre_counts.index)}
coded_genres_reverse = {k: genre for genre, k in coded_genres.items()}

print(coded_genres)

# X frames with 50% overlap = 2X-1 frames
num_frames = 4
total_frames = 2 * num_frames - 1
frame_size = 1290 // num_frames

class FeatureDataset(Dataset):
    def __init__(self, agfs=[], genre=True):
        super().__init__()
        self.agfs = agfs
        self.genre = genre
        
    def __len__(self):
        return len(target_paths)
    
    def __getitem__(self, idx):
        path = target_paths[idx]
        tid = tids[idx]
        
        # argmax these
        features = {}
        
        with np.load(path) as data:
            for agf in self.agfs:
                lda_vec = data[agf]
                features[agf] = lda_vec.argmax() #torch.from_numpy((lda_vec == lda_vec.max())*1) #argmax the output of Latent Dirichlet Allocation
            
            mel = data['mel']
        
        if self.genre:
            features['genre'] = coded_genres[tracks_subset['genre_top'][tid]]
        
        return mel, features

class FramedFeatureDataset(FeatureDataset):
    def __len__(self):
        return total_frames * super().__len__()
    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        
        song_idx, frame = divmod(idx, total_frames)
        mel, features = super().__getitem__(song_idx)
        
        shift, half_shift = divmod(frame, 2)
        i = shift * frame_size + half_shift * frame_size // 2
        
        # add channel dimension so its 1x128x(frame_size)
        mel_frame = np.expand_dims(mel[:, i:i + frame_size], axis=0)
        
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
