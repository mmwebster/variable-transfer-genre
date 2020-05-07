import numpy as np
import glob
import os
from itertools import compress, product
from pathlib import Path

# copy copy
def combinations(items):
    return ( set(compress(items,mask)) for mask in product(*[[0,1]]*len(items)) )

def list_to_dict(the_list):
    return {key : idx for idx, key in enumerate(the_list)}

# hack way to get around mis-ordering of combinations(items) w.r.t to order of items
def sort_targets(targets_to_sort, targets_dict):
    indexes = [targets_dict[name] for name in targets_to_sort]
    sorted_indexes = sorted(range(len(indexes)), key=lambda k: indexes[k])
    return [targets_to_sort[i] for i in sorted_indexes]

def load_available_f1s(dataset_name, possible_targets, possible_layers):
    # list of dictionaries of final F1 scores for MLPs transferred from various combinations of STNs
    f1s = []
    possible_targets_dict = list_to_dict(possible_targets)

    # iterate through all combinations w/ stns ordered as above and possible layers
    # just try/except to take the existing files
    # filename = "logs/final_MLP_fma_medium_stn_subgenres_layer_7"
    # final_f1 = np.fromfile(filename)
    # print(f'final: {final_f1}')
    for targets in combinations(possible_targets):
        sorted_targets = sort_targets(list(targets), possible_targets_dict)
        # @TODO: same thing as for targets, for layers (we'll try different layer combinations across the different STNs)
        for layer in possible_layers:
            final_f1 = "NA"
            filename = f'logs/final_MLP_{dataset_name}_stn_{"_".join(sorted_targets)}_layer_{layer}'
            # load the metric if it is available
            if Path(filename).is_file():
                f1 = np.fromfile(filename)
                f1s.append({"targets": sorted_targets, "layer": layer, "f1": f1})
    return f1s
