import os
import numpy as np
import glob
import torch
import warnings
from shutil import copyfile

##
# @brief One-off utility to copy unique track data from small dataset to
#        a test set for medium
#
# About 160 tracks in small are not contained in medium, and so since
#   - we trained on medium
#   - we couldn't store the full (large) fma dataset on our cloud VM instance
#   - we weren't entirely sure we didn't mix training and validation data
#   - and didn't have time to retrain models to verify this
# we extracted a 'test' set from fma_small's unique tracks on which our model
# should perform roughly as well as for our validation set on fma_medium
##

DATA_REL_PATH_MED = '../data/fma_medium/'
DATA_REL_PATH_S = '../data/fma_small/'
DATA_REL_PATH_MED_TEST = '../data/fma_medium_testset/'

def get_dataset_tids(path):
    # path to FMA audio data
    audio_dir = os.path.join(os.path.curdir, path)
    # path to 'raw' extracted features
    raw_dir = os.path.join(audio_dir, 'raw')

    raw_paths = [*glob.iglob(os.path.join(raw_dir, '*.npz'), recursive=True)]
    tids = list(map(lambda x: int(os.path.splitext(os.path.basename(x).replace('_raw', ''))[0]), raw_paths))

    return tids

def get_lhs_unique_tids(small_tids, med_tids):
    tid_dict = {}
    for tid in med_tids:
        tid_dict[str(tid)] = 1

    small_unique_tids = []
    for tid in small_tids:
        if not str(tid) in tid_dict:
            small_unique_tids.append(tid)

    print(f"fma_small has {len(small_unique_tids)} TIDs that aren't in fma_medium small")
    
    return small_unique_tids

# copy 'raw' and 'target' data for these unique TIDs into a fma_medium_testset
def copy_unique_track_data(tids, from_dir, to_dir):
    for tid in tids:
        # copy 'raw' data
        src = f"{from_dir}raw/{tid}_raw.npz"
        dst = f"{to_dir}raw/{tid}_raw.npz"
        copyfile(src, dst)
        
        # copy 'target' data
        src = f"{from_dir}targets/{tid}_targets.npz"
        dst = f"{to_dir}targets/{tid}_targets.npz"
        copyfile(src, dst)
        
if __name__ == "__main__":
    # get dataset tids
    med_tids = get_dataset_tids(DATA_REL_PATH_MED)
    small_tids = get_dataset_tids(DATA_REL_PATH_S)
    
    # get unique tids in small
    small_unique_tids = get_lhs_unique_tids(small_tids, med_tids)
    
    # copy small's unique track data to a testset for medium
    copy_unique_track_data(small_unique_tids, DATA_REL_PATH_S, DATA_REL_PATH_MED_TEST)
    