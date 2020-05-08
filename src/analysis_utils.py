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

# return string of target first letters for each target in targets
def abbrev_targets(target_names):
    name = ""
    for target_name in target_names:
        name += target_name[0]
    return name.upper()

# Returns metadata for MLP models that have been trainined and are available for analysis
# ** hacky way of gathering model metadata from saved model data filenames **
# available_mlp_models = [
#   {
#       'targets': [targets]
#       'layer': [layer]
#   }
# ]
def find_available_mlp_models(dataset_name, possible_targets, possible_layers):
    available_mlp_models = []
    possible_targets_dict = list_to_dict(possible_targets)

    # iterate through all combinations of targets and layers, ignoring those not present
    for targets in combinations(possible_targets):
        sorted_targets = sort_targets(list(targets), possible_targets_dict)
        # @TODO: same thing as for targets, for layers (we'll try different layer combinations across the different STNs)
        for layer in possible_layers:
            filename = f'logs/final_MLP_{dataset_name}_stn_{"_".join(sorted_targets)}_layer_{layer}'
            # save mlp model metadata if it is available
            if Path(filename).is_file():
                available_mlp_models.append({"dataset": dataset_name, "targets": sorted_targets, "layer": layer})
    return available_mlp_models

# Take a list of MLP model metadata and return list of corresponding final F1 scores for each model
def get_mlp_f1s(available_mlp_models):
    # list of dictionaries of final F1 scores for MLPs transferred from various combinations of STNs
    f1s = []

    # iterate through all combinations of targets and layers, ignoring those not present
    for model in available_mlp_models:
        filename = f"logs/final_MLP_{model['dataset']}_stn_{'_'.join(model['targets'])}_layer_{model['layer']}"
        f1_one_element_array = np.fromfile(filename)
        f1s.append(f1_one_element_array[0])

    return f1s

# Take a list of MLP model metadata and return list of corresponding loss histories for each model
def get_mlp_loss_histories(available_mlp_models):
    # list of dictionaries of loss histories for MLPs transferred from various combinations of STNs
    loss_histories = []

    # iterate through all combinations of targets and layers, ignoring those not present
    for model in available_mlp_models:
        filename = f"logs/losses_MLP_{model['dataset']}_stn_{'_'.join(model['targets'])}_layer_{model['layer']}"
        loss_histories.append(np.fromfile(filename))

    return loss_histories

# Take a list of MLP model metadata and return list of corresponding final F1 scores for each model
def get_stn_f1s(available_stn_models):
    # list of dictionaries of final F1 scores for MLPs transferred from various combinations of STNs
    f1s = []

    # iterate through all combinations of targets and layers, ignoring those not present
    for model in available_stn_models:
        filename = f"logs/final_STN_{model['dataset']}_{model['target']}"
        f1_one_element_array = np.fromfile(filename)
        f1s.append(f1_one_element_array[0])

    return f1s
