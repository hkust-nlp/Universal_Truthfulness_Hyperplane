import torch
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from tqdm import tqdm
from utils import load_model, get_activations_bau, load_dataset, open_json_file
import argparse
import numpy as np
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
def batch_process(data, batch_size):
    num_batches = (len(data) + batch_size - 1) // batch_size
    batches = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batch = data[start_idx:end_idx]
        batches.append(batch)

    return batches

def main(model_name_path:str=None, posi:int=-1,data_name:str='saplma',save_folder:str=None):
    model, tokenizer = load_model(model_name_path, device)
    model.eval()
    dataset = load_dataset(dataset_name=data_name, prompt_format=args.prompt_format)
    """ prompt_format: arc_qa_prompt   or   general_qa_prompt
    """
    print(data_name+"  "+str(len(dataset['data'])))
    labels = dataset['labels']
    prompt_list = dataset['data']
    print(f"here is the example:\n{prompt_list[0]}")
    bsz = 100
    bsz_data = batch_process(prompt_list, bsz)
    bsz_labels = batch_process(labels, bsz)
    for round_index in range(len(bsz_data)):
        print(f"----------round: {round_index+1}/{len(bsz_data)} --------------")
        prompt_list = bsz_data[round_index]
        labels = bsz_labels[round_index]
        all_layer_hidden_states = []
        all_head_hidden_states = []
        all_mlp_hidden_states = []
        all_mid_mlp_hidden_states = []
        i=0
        for prompt in tqdm(prompt_list):
            # print(prompt)
            input_ids = tokenizer(prompt,return_tensors='pt',padding=True, max_length=4096).input_ids
            # print(input_ids)
            num_heads = model.config.num_attention_heads
            # print(f"num of heads {num_heads}")
            hidden_states_res = get_activations_bau(model=model,prompt=input_ids,device=device, is_Head=True, is_Layer=True, is_MLP=False, is_MID_MLP=False, only_last=True)
            # all the three are shape layers x tokens x hidden_dim
            # print(hidden_states.shape)
            # print(head_wise_hidden_states.shape)
            # print(mlp_wise_hidden_states.shape)

            all_layer_hidden_states.append(hidden_states_res['layer'])
            all_head_hidden_states.append(hidden_states_res['head'])
            # all_mlp_hidden_states.append(mlp_wise_hidden_states[:,-1,:])
            # all_mid_mlp_hidden_states.append(mid_mlp_wise_hidden_states[:,-1,:])
            i+=1
            
        train_str='_probe_train'
        vali_str='_probe_vali'
        file_name_list = []
        task_name_list = ['capitals','companies','elements','facts','inventions', 'animals','easy_arc', 'arc', 'tqa', 'nq_iti', 'triva_qa_iti', 'triva_qa_re', 'nq_re', 'counterfact', 'openbookqa', 'ag_news', 'imdb', 'boolq', 'piqa',
                            "arithmetic", "commonsense_qa", "copa", "rte", "sciq", "creak", "qnli", "strategy_qa", "mrpc", 'neg_facts', 'neg_companies', 'triva_qa_re_long', 'nq_re_long',
                            'hellaswag', 'race', 'quartz', 'dream', 'quarel', 'yelp_polarity', 'qqp', 'dbpedia_14',  'winogrande', 'anli', 'xsum_re', 'cnn_dailymail_re',
                            'paws','multirc','squad','e2e_nlg_cleaned','web_nlg_re','fr-en','de-en','definite_pronoun_resolution','wsc.fixed','record','cosmos_qa', 'hotpot_qa_re', 'wic', 'story_cloze']
        for task in task_name_list:
            file_name_list.append(task+train_str)
            file_name_list.append(task+vali_str)
            
        dataset_name = ""
        for temp_name in file_name_list:
            if temp_name in data_name:
                dataset_name=temp_name
                break
        assert dataset_name!="", "wrong definition of the data name"

        print(len(all_mid_mlp_hidden_states))
        print("Saving labels")

        print(dataset_name)
        prefix_path = args.prefix_path
        np.save(f'{prefix_path}/{save_folder}/{args.model_name}_{dataset_name}_labels_{round_index}.npy', labels)

        print("Saving layer wise activations")
        np.save(f'{prefix_path}/{save_folder}/{args.model_name}_{dataset_name}_layer_wise_{round_index}.npy', all_layer_hidden_states)
        
        print("Saving head wise activations")
        np.save(f'{prefix_path}/{save_folder}/{args.model_name}_{dataset_name}_head_wise_{round_index}.npy', all_head_hidden_states)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-m','--model_path',type=str,default="NousResearch/Llama-2-7b-hf",help="base model to use")
    parser.add_argument('--model_name',type=str,default="llama2",help="base model to use")
    parser.add_argument('--data_name_path',type=str,default='saplma')
    parser.add_argument('--save_folder',type=str)
    parser.add_argument('--prefix_path',type=str,default='.')
    args=parser.parse_args()
    print(args)
    main(model_name_path=args.model_path, data_name=args.data_name_path, save_folder=args.save_folder)
   