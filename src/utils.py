from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import Trace, TraceDict
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
import random
import torch as t
import pandas as pd
from sklearn.decomposition import PCA
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

general_qa_prompt="""Question: {}
Answer: {}"""

def open_json_file(file_name):
    if '/data' not in file_name:
        file_name = os.path.join('./data',file_name)
    with open(file_name, "r") as file:
        data = json.load(file)
        return data
def convert_score2acc(score,label):
    binary_pred = (score >= 0.5).astype(int)
    accuracy = np.mean(binary_pred == label)
    return accuracy
def sigmoid(x):
    x_ravel = x.ravel()  # 将numpy数组展平
    length = len(x_ravel)
    y = []
    for index in range(length):
        if x_ravel[index] >= 0:
            y.append(1.0 / (1 + np.exp(-x_ravel[index])))
        else:
            y.append(np.exp(x_ravel[index]) / (np.exp(x_ravel[index]) + 1))
    return np.array(y).reshape(x.shape)

def load_model(model_name, device):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    print(f"putting model to {device} ")
    return model, tokenizer

def load_train_data(model_name, dataset_name, posi, upper_bound=-1, num_heads=32):
    if model_name == 'llama2' or model_name == 'llama2_7b' or model_name == 'llama2_7b_chat' or model_name == 'mistral_7b':
        assert num_heads==32, "wrong head num"
    elif model_name == 'llama2_13b' or model_name == 'llama2_13b_chat':
        assert num_heads==40, "wrong head num"
        
    name = model_name+'_'+dataset_name+'_'+posi+'.npy'
    label = load_npy(file_name=model_name+'_'+dataset_name+'_labels.npy')
    print("train name "+name)
    
    data = load_npy(file_name=name)
    if posi == 'head_wise':
        data = split_head_states(data, num_heads=num_heads, all_tokens=False)
    if upper_bound != -1:
        print(f"load train num: {upper_bound}")
        data = data[:upper_bound]
        label =  label[:upper_bound]
    return data, label

def load_test_data(model_name,  test_file=None, posi=None, num_heads=32, selected_vali=False, selected_test=False):
    if model_name != 'llama2' and model_name != 'llama2_7b_chat' and model_name != 'mistral_7b'  and model_name != 'llama2_7b_finetuned' and num_heads == 32:
        assert False, "Please check whether the num_head is correct"
    if model_name == 'llama2' or model_name == 'llama2_7b' or model_name == 'llama2_7b_chat' or model_name=='mistral_7b' or model_name == 'llama2_7b_finetuned':
        assert num_heads==32, "wrong head num"
    elif model_name == 'llama2_13b' or model_name == 'llama2_13b_chat':
        assert num_heads==40, "wrong head num"

    name = model_name+'_'+test_file+'_'+posi+'.npy'
    label = load_npy(file_name=model_name+'_'+test_file+'_labels.npy')
    print("test name "+name)
    
    data = load_npy(file_name=name)
    if posi == 'head_wise':
        data = split_head_states(data, num_heads=num_heads, all_tokens=False)

    if selected_vali == True:
        data = data[:100]
        label = label[:100]
    elif selected_test == True:
        data = data[100:]
        label = label[100:]
    return data, label

def get_activations_bau(model, prompt, device, is_Head=False, is_Layer=False, is_MLP=False, is_MID_MLP=False, is_EXPERT=False, only_last=True): 

    model.eval()

    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    MID_MLPS = [f"model.layers.{i}.mlp.down_proj" for i in range(model.config.num_hidden_layers)]
    if is_EXPERT == True:
        EXPERTS = [f"model.layers.{i}.block_sparse_moe.experts.{j}"  for i in range(model.config.num_hidden_layers) for j in range(model.config.num_experts)]
    selected = []
    if is_Head:
        selected += HEADS
    if is_MLP:
        selected += MLPS
    if is_MID_MLP:
        selected += MID_MLPS
    if is_EXPERT:
        selected += EXPERTS
    results ={}
    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, selected, retain_input=True) as ret:
            output = model(prompt, output_hidden_states = is_Layer)
        if is_Layer:
            hidden_states = output.hidden_states
            hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
            hidden_states = hidden_states.detach().cpu().numpy()
            if only_last:
                hidden_states = hidden_states[:,-1,:]
            results['layer'] = hidden_states
        if is_Head:
            head_wise_hidden_states = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
            head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
            if only_last:
                head_wise_hidden_states = head_wise_hidden_states[:,-1,:]
            results['head'] = head_wise_hidden_states
        if is_MLP:
            mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
            mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()
            if only_last:
                mlp_wise_hidden_states = mlp_wise_hidden_states[:,-1,:]
            results['mlp'] = mlp_wise_hidden_states
        if is_MID_MLP:
            mid_mlp_wise_hidden_states = [ret[mlp].input.squeeze().detach().cpu() for mlp in MID_MLPS]
            mid_mlp_wise_hidden_states = torch.stack(mid_mlp_wise_hidden_states, dim = 0).squeeze().numpy()
            if only_last:
                mid_mlp_wise_hidden_states = mid_mlp_wise_hidden_states[:,-1,:]
            results['mid_mlp'] = mid_mlp_wise_hidden_states
        if is_EXPERT:
            expert_wise_hidden_states = [ret[expert].output.squeeze().detach().cpu() for expert in EXPERTS]
            expert_wise_hidden_states = torch.stack(expert_wise_hidden_states, dim = 0).squeeze().numpy()
            if only_last:
                expert_wise_hidden_states = expert_wise_hidden_states[:,-1,:]
            results['expert'] = expert_wise_hidden_states
    return results

def split_head_states(head_hidden_states, num_heads, all_tokens=True):
    if all_tokens == True:
        """
        input: head_hidden_states: shape  layers x tokens x head_num*head_dim

        aim: split it into different heads

        return:  tensor shapes like    layers x tokens x head_num x head_dim
        """
        if len(head_hidden_states.shape) == 3:
            layers_num, tokens_num, head_mul_head_dim = head_hidden_states.shape
            split_head_hidden_states = head_hidden_states.reshape(layers_num, tokens_num, num_heads, head_mul_head_dim//num_heads)
            return split_head_hidden_states
        elif len(head_hidden_states.shape) == 4:
            bsz, layers_num, tokens_num, head_mul_head_dim = head_hidden_states.shape
            split_head_hidden_states = head_hidden_states.reshape(bsz, layers_num, tokens_num, num_heads, head_mul_head_dim//num_heads)
            return split_head_hidden_states
        else:
            dim = len(head_hidden_states.shape)
            assert False, f"Wrong dimension of input head_hidden_states, require 3 or 4, but input is {dim}"
    else:
        """
        input: bsz x layers x head_num*head_dim
        return: bsz x layers x head_num x head_dim
        """
        if len(head_hidden_states.shape) == 2:
            layers_num, head_mul_head_dim = head_hidden_states.shape
            split_head_hidden_states = head_hidden_states.reshape(layers_num, num_heads, head_mul_head_dim//num_heads)
            return split_head_hidden_states
        elif len(head_hidden_states.shape) == 3:
            bsz, layers_num, head_mul_head_dim = head_hidden_states.shape
            split_head_hidden_states = head_hidden_states.reshape(bsz, layers_num, num_heads, head_mul_head_dim//num_heads)
            return split_head_hidden_states
        else:
            dim = len(head_hidden_states.shape)
            assert False, f"Wrong dimension of input head_hidden_states, require 3 or 4, but input is {dim}"

def load_dataset(dataset_name: str=None, seed:int =0, prompt_format:str =None):
    """ prompt_format: indicate what prompt format we use
        return  {'data':data_str, 'labels':data_label}
    """
    random.seed(seed)
    if '.json' in dataset_name:
        
        data = open_json_file(dataset_name)
        data_str = []
        data_label = []
        if 'question' in data[0].keys() and 'answer' in data[0].keys():
            # here we use the qa format

            used_prompt = general_qa_prompt
            print("----We are using the format "+used_prompt)
            for d in data:
                q = d['question']
                a = d['answer']
                temp_str = used_prompt.format(q,a)
                data_str.append(temp_str)
                data_label.append(d['label'])
            res = {'data':data_str, 'labels':data_label}
            return res
        elif 'data' in data[0].keys():
            for d in data:
                data_str.append(d['data'])
                data_label.append(d['label'])
            res = {'data':data_str, 'labels':data_label}
            return res
        else:
            assert False, f"No define of the key {data[0].keys()}"

def load_npy(file_name, features_folder):
    
    file_path = ""

    task1_file_list = ['ag_news', 'cnn_dailymail_re', 'dbpedia_14', 'facts', 'nq_re_long', 'race','triva_qa_re_long', 'triva_qa_re', 'animals', 'commonsense_qa', 'de-en', 'fr-en', 
    'openbookqa', 'record',  'anli', 'companies', 'definite_pronoun_resolution', 'hellaswag', 'paws', 'rte', 'web_nlg_re', 'easy_arc',  'arc', 'copa', 'dream',
    'hotpot_qa_re', 'piqa', 'sciq', 'winogrande', 'arithmetic', 'cosmos_qa', 'e2e_nlg_cleaned', 'inventions', 'qnli', 'squad', 'wsc.fixed', 'boolq', 'counterfact', 
    'multirc', 'qqp', 'strategy_qa', 'xsum_re', 'capitals', 'creak', 'elements', 'nq_re', 'quarel', 'tqa', 'yelp_polarity', 'imdb', 'mrpc', "story_cloze", "wic"]
    
    for temp_task in task1_file_list:
        if temp_task in file_name:
            file_path = os.path.join(f"{features_folder}/{temp_task}", file_name)
            break

    assert file_path!="", f"cannot find the folder for file {file_path}"
    
    assert os.path.exists(file_path), f"There is no file named {file_path}"
    data = np.load(file_path)
    return data


def split_train_validation(data, labels, portion):
    """ portion is a number between 0 and 1
    """
    if type(data) == np.ndarray:
        all_num = data.shape[0]
    elif type(data) == list:
        all_num = len(data)
    else:
        data_type = type(data)
        assert False, f"wrong data type, must be np.ndarray or list but given {data_type}"
    train_num = round(all_num*portion)

    train_data = data[:train_num]
    train_label = labels[:train_num]

    vali_data = data[train_num:]
    vali_label = labels[train_num:]
    return train_data, train_label, vali_data, vali_label

def sync_shuffle(data, label):
    assert data.shape[0]==label.shape[0],"mismatch of the train data num and the label num"
    data_num = data.shape[0]
    random_indices = np.random.permutation(data_num)

    data = data[random_indices]
    label = label[random_indices]
    return data,label








