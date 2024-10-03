from argparse import ArgumentParser
from pathlib import Path
import pickle
import numpy as np
import random
import itertools
import sys
import numpy as np
from utils import   load_npy, split_head_states, load_train_data, load_test_data,  sync_shuffle
from probe_utils import get_probe_acc, train_single_probe,  test_probe_data, MMProbe, MLP_Probe
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from into_csv import write_csv
from itertools import permutations
import json
from matplotlib import pyplot

def has_lr_model(train_task_name_list, posi):
    if isinstance(train_task_name_list, str):
        train_task_name_list = [train_task_name_list]
    permutations_list = list(permutations(train_task_name_list))
    file_name_pefix_list = []
    for temp in permutations_list:
        file_name_pefix_list.append('_'.join(temp)+'_'+posi+'.pkl')
    for file_name in file_name_pefix_list:
        file_path = os.path.join('./lr_models', file_name) 

        if os.path.isfile(file_path):
            return True, file_path
    return False,""


def extract_w(probe):
    if isinstance(probe, LogisticRegression):
        weights = probe.coef_
        # print(weights)
        # print(weights.shape)
        return weights
    elif isinstance(probe, MMProbe):
        weights = probe.direction
        weights = weights.cpu().numpy()
        weights =  weights.reshape(1,-1)
        return weights
    
def get_dim(probe):
    if isinstance(probe, list):
        probe = probe[0]
    w = extract_w(probe)
    return w.shape[1]

def w_find_k_largest(w, k):
    """
    finding the largest k weight position to suppress the hidden state.
    w: (1,dim)
    k: the largest k num
    
    return: the index of the k largest number
    """
    abs_w = np.abs(w)
    indices = np.argsort(abs_w)[0, -k:] 
    nonzero_indices = indices[np.nonzero(w[0, indices])]
    if indices.shape != nonzero_indices.shape:
        return indices
    return nonzero_indices

def is_compressible(probe, k):
    """
    is compressible if and only if k <= probe dim
    """
    w = extract_w(probe)
    return k <= w.shape[1]

def compress_probe(probe,k=1):
    """
    probe:  the logistic regression trained by sklearn or a probe list
    k: k largest number to return
    """
    
    if not isinstance(probe, list):
        probe = [probe]
    ans_indices=[]
    for p in probe:
        w = extract_w(probe=p)
        indices = w_find_k_largest(w=w, k=k)
        # breakpoint()
        ans_indices.append(indices)
    return ans_indices
    
def probeless(data, label):
    data_0 = data[label == 0]
    data_1 = data[label == 1]
    mean_0 = np.mean(data_0, axis=0)
    mean_1 = np.mean(data_1, axis=0)
    scores = np.abs(mean_0-mean_1)
    return scores

def select_index(acc, selected_num, max_diff_acc):
    if isinstance(acc, list):
        acc = np.array(acc)
    sorted_indices = np.argsort(acc)
    acc_max = np.max(acc)
    if selected_num == -1:
        indices = sorted_indices
    else:
        indices = sorted_indices[-selected_num:]
        threshold = acc_max - max_diff_acc
        indices = indices[np.where(acc[indices] >= threshold)]
    print(f"selected acc {acc[indices]}")
    print(f"selected indices {indices}")
    print(f"selected num {indices.shape[0]}")
    return indices

def pure_merge_dataset(model_name, dataset_name, posi,  num_heads=32,  test_file=None,
                       selected_vali=False, selected_test=False):
    other_train_data = []
    other_train_label = [] 

    
    if dataset_name!=None and isinstance(dataset_name,str):
        dataset_name = [dataset_name]
    if dataset_name is not None:
        for other_dataset in dataset_name:
            temp_data, temp_label = load_train_data(model_name=model_name, dataset_name=other_dataset, posi=posi, 
                                 num_heads=num_heads)
            other_train_data.append(temp_data)
            other_train_label.append(temp_label)
    # collect training dataset done!
    if other_train_data!=[]:
        train_data = np.concatenate(other_train_data, axis=0)
        train_label = np.concatenate(other_train_label, axis=0)
        train_data, train_label = sync_shuffle(train_data, train_label)
        
    
    vali_data, vali_label = load_test_data(model_name=model_name, test_file=test_file, posi=posi, num_heads=num_heads, selected_vali=selected_vali, selected_test=selected_test)
    if  posi=='head_wise':
        bsz,_,_,head_dim = train_data.shape
        train_data = train_data.reshape(bsz, -1, head_dim)
        vali_bsz, _,_, head_dim = vali_data.shape
        vali_data = vali_data.reshape(vali_bsz, -1, head_dim)
    return train_data, train_label, vali_data, vali_label

def selective_merge_position_data(  model_name, dataset_name, posi, selected_num=1, posi_list=None, acc=None, indices=None, num_heads=32, test_file=None, max_diff_acc=1, selected_flat_indices=None
                        , selected_vali=False, selected_test=False):
    train_data, train_label, vali_data, vali_label = pure_merge_dataset(model_name=model_name, dataset_name=dataset_name, posi=posi, num_heads=num_heads, test_file=test_file,
                                                                        selected_vali=selected_vali, selected_test=selected_test)
    if selected_flat_indices is None:
        # according to acc select `selected_num` positions to use
        if indices is None:
            indices = select_index(acc, selected_num, max_diff_acc)
        merged_data_list = []
        vali_merged_data_list = []
        for ind in indices:
            merged_data_list.append( train_data[:,ind,posi_list[ind]] )
            vali_merged_data_list.append( vali_data[:,ind, posi_list[ind]])
        merged_data = np.concatenate(merged_data_list, axis=1)
        vali_merged_data = np.concatenate(vali_merged_data_list, axis=1)

        return merged_data, train_label, vali_merged_data, vali_label
    else:
        train_bsz = train_data.shape[0]
        vali_bsz =  vali_data.shape[0]
        train_data = train_data.reshape(train_bsz, -1)
        vali_data = vali_data.reshape(vali_bsz, -1)
        train_data = train_data[:,selected_flat_indices]
        vali_data = vali_data[:,selected_flat_indices]
        
        return train_data, train_label, vali_data, vali_label

    
model_name_list = ['llama2','llama2_13b']
dataset_name_list = ['saplma']
topic_name_list = ['animals','capitals','companies','elements','facts','inventions']
model_posi = ['head_wise','layer_wise','mlp_wise','mid_mlp_wise']

def compress(k=1, selected_num=1, posi='head_wise',  model_name=None,dataset_name=None, test_file=None,  is_MLP=False, is_MM=False, num_heads=32, exit_posi=False, max_diff_acc=1, solver=None, penalty=None, is_probe=True, input_dim=100,
             selected_vali=False, selected_test=False):
    if model_name == 'llama2_13b_chat':
        num_heads = 40
    if is_probe == True:
        old_probs,acc = get_probe_acc(model_name=model_name, dataset_name=dataset_name,  posi=posi,  test_file=test_file,  is_MLP=is_MLP, is_MM=is_MM, num_heads=num_heads, solver=solver, penalty=penalty, selected_vali=selected_vali, selected_test=selected_test)
        max_index = np.argmax(acc)
        print(f"before acc {acc[max_index]}")
        old_acc = acc[max_index]
        posi_list = compress_probe(old_probs,  k=k )
        if exit_posi:
            return posi_list,acc, old_probs
        print(f" compressing {posi} neurons to {k}")
        # print(f"using {selected_num} positions to train")
        data, label, vali_data, vali_label = selective_merge_position_data(acc=acc, posi_list=posi_list, selected_num=selected_num, model_name=model_name, dataset_name=dataset_name, posi=posi, num_heads=num_heads, test_file=test_file, max_diff_acc=max_diff_acc,
                                                                selected_vali=selected_vali, selected_test=selected_test)
        probe, acc = train_single_probe(x_train=data, y_train=label, x_val=vali_data, y_val=vali_label, is_MM=is_MM, is_MLP=is_MLP)
        # print( "acc")
        return probe, acc
    else:
        train_data, train_label, vali_data, vali_label = pure_merge_dataset(model_name=model_name, dataset_name=dataset_name, posi=posi,  num_heads=num_heads,  test_file=test_file,
                                                                            selected_vali=selected_vali, selected_test=selected_test)
        scores = probeless(data = train_data, label = train_label)  # shape like position x dim
        # breakpoint()
        print(f"selecting {input_dim} neurons")
        selected_flat_indices = np.argsort(scores.flatten())[::-1][:input_dim]
        if exit_posi:
            return selected_flat_indices
        data, label, vali_data, vali_label = selective_merge_position_data( model_name=model_name, dataset_name=dataset_name, posi=posi, num_heads=num_heads, test_file=test_file, selected_flat_indices=selected_flat_indices,
                                                                 selected_vali=selected_vali, selected_test=selected_test)
        probe, acc = train_single_probe(x_train=data, y_train=label, x_val=vali_data, y_val=vali_label, is_MM=is_MM, is_MLP=is_MLP)
        return probe, acc

def specify_classifer(k, model_name=None,dataset_name=None, topic_name=None, test_topic=None, test_file=None,  is_MLP=False, num_heads=32, posi_index=-1, posi=None):
    old_probs,acc = get_probe_acc(model_name=model_name, dataset_name=dataset_name, topic_name=topic_name, posi=posi, portion=0.7, test_file=test_file, test_topic=test_topic, is_MLP=is_MLP, num_heads=num_heads)
    indices = compress_probe(old_probs[posi_index], k=k)
    print(f"acc in position {posi_index} is: {acc[posi_index]}")
    print(indices)
    return indices

def merge_indices(indices_list):
    if isinstance(indices_list, list):
        indices_list = np.array(indices_list)
    indices_list = np.concatenate(indices_list)
    indices = np.unique(indices_list)
    return indices
