import numpy as np
import glob
import os
from itertools import compress, product
from pathlib import Path

# stack @ninjagecko
def combinations(items):
    return ( set(compress(items,mask)) for mask in product(*[[0,1]]*len(items)) )

def list_to_dict(the_list):
    return {key : idx for idx, key in enumerate(the_list)}

# hacky way to get around output unordering of combinations(items) w.r.t to input ordering
def sort_targets(targets_to_sort, targets_dict):
    indexes = [targets_dict[name] for name in targets_to_sort]
    sorted_indexes = sorted(range(len(indexes)), key=lambda k: indexes[k])
    return [targets_to_sort[i] for i in sorted_indexes]

def load_available_f1s(dataset_name, possible_targets, possible_layers):
    # list of dictionaries of final F1 scores for MLPs transferred from various combinations of STNs
    f1s = []
    possible_targets_dict = list_to_dict(possible_targets)

    # iterate through all combinations of targets and layers, ignoring those not present
    for targets in combinations(possible_targets):
        sorted_targets = sort_targets(list(targets), possible_targets_dict)
        # @TODO: same thing as for targets, for layers (we'll try different layer combinations across the different STNs)
        for layer in possible_layers:
            filename = f'logs/final_MLP_{dataset_name}_stn_{"_".join(sorted_targets)}_layer_{layer}'
            # load the metric if it is available
            if Path(filename).is_file():
                f1 = np.fromfile(filename)
                f1s.append({"targets": sorted_targets, "layer": layer, "f1": f1})
    return f1s

def load_available_loss_histories(dataset_name, possible_targets, possible_layers):
    loss_histories = []
    possible_targets_dict = list_to_dict(possible_targets)

    # iterate through all combinations of targets and layers, ignoring those not present
    for targets in combinations(possible_targets):
        sorted_targets = sort_targets(list(targets), possible_targets_dict)
        # @TODO: same thing as for targets, for layers (we'll try different layer combinations across the different STNs)
        for layer in possible_layers:
            filename = f'logs/losses_MLP_{dataset_name}_stn_{"_".join(sorted_targets)}_layer_{layer}'
            # load the metric if it is available
            if Path(filename).is_file():
                loss_history = np.fromfile(filename)
                loss_histories.append({"targets": sorted_targets, "layer": layer, "loss_history": loss_history})

    return loss_histories
