import csv
import os
columns_list = ['capitals','companies','elements','facts','inventions', 'animals', 'tqa', 'nq_re', 'triva_qa_re', 
                "ag_news",  "arc",  "arithmetic",  "boolq",  "commonsense_qa",  "copa",  "counterfact",  "imdb",  "mrpc",  "neg_companies",  "neg_facts",
                "nq_re_long",  "piqa",  "qnli",  "rte",  "sciq",  "strategy_qa",  "triva_qa_re_long", "creak",  "hellaswag",  "race", "openbookqa",
                'dbpedia_14',  'dream',  'qqp',  'quarel',  'quartz',  'yelp_polarity', 'easy_arc',  'winogrande', 'anli', 'xsum_re', 'cnn_dailymail_re',
                 'paws','multirc','squad','e2e_nlg_cleaned','web_nlg_re','fr-en','de-en','definite_pronoun_resolution','wsc.fixed','record','cosmos_qa', 'hotpot_qa_re',"story_cloze", "wic"]
model_posi = ['head_wise','layer_wise']
# ,'mlp_wise','mid_mlp_wise'
def get_average(head_row):
    sum_acc = 0
    count = 0
    for i in head_row:
        if not isinstance(i, str):
            sum_acc += i
            count+=1
    return sum_acc/count
def match_check(test_name, columns_list = columns_list):
    for i in range(len(columns_list)):
        if columns_list[i] == test_name or columns_list[i]+'_probe_vali' == test_name or columns_list[i]+'_probe_vali_Arc' == test_name or columns_list[i]+'_Statement' == test_name:
            return i+1
    return -1
def write_csv(ans, file_name, title):
    file_name = os.path.join('csv_result', file_name)
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        row1 = [title]+columns_list
        head_row = ['']*len(row1)
        head_row[0] = 'head_wise'
        layer_row = ['']*len(row1)
        layer_row[0] = 'layer_wise'
        mlp_row = ['']*len(row1)
        mlp_row[0] = 'mlp_wise'
        mid_mlp_row = ['']*len(row1)
        mid_mlp_row[0] = 'mid_mlp_wise'
        if 'Mean'  in ans[0].keys():
            writer.writerow(row1+['average','posi'])
            posi = ans[0]['posi_list']
            for i in range(len(ans)):
                test_item = ans[i]['name'].split('test on ')[1].strip()
                index = match_check(test_name=test_item)
                acc_list = ans[i]['acc_list']
                head_row[index] = acc_list[0]
                layer_row[index] = acc_list[1]
                # mlp_row[index] = acc_list[2]
                # mid_mlp_row[index] = acc_list[3]
            head_avg = get_average(head_row)
            layer_avg = get_average(layer_row)
            # mlp_avg = get_average(mlp_row)
            # mid_avg = get_average(mid_mlp_row)
            #, mlp_row+[mlp_avg, posi[2]], mid_mlp_row+[mid_avg, posi[3]]
            sum_rows = [head_row+[head_avg, posi[0]],layer_row+[layer_avg, posi[1]]]
            writer.writerows(sum_rows)
            print(f"save done!  {file_name}")
        elif 'compress' in ans[0].keys():
            new_row = row1+['average','k','num','input dim', 'all num']
            writer.writerow(new_row)
            
            for a in ans:
                head_row = ['']*len(new_row)
                for temp_column in a.keys():
                    print(f"current column: {temp_column}")
                    index = match_check(temp_column, new_row)-1
                    if index <0 :
                        print(f"Can't find {temp_column}")
                        continue
                    head_row[index] = a[temp_column]
                    if temp_column == 'input dim':
                        print(head_row[index])
                print(head_row)
                writer.writerow(head_row)
            print(f"save done!  {file_name}")
        else:
            writer.writerow(row1+['average'])
            for i in range(len(ans)):
                test_item = ans[i]['name'].split('test on ')[1]
                index = match_check(test_name=test_item)
                acc_list = ans[i]['acc_list']
                head_row[index] = acc_list[0]
                layer_row[index] = acc_list[1]
                # mlp_row[index] = acc_list[2]
                # mid_mlp_row[index] = acc_list[3]
            head_avg = get_average(head_row)
            layer_avg = get_average(layer_row)
            # mlp_avg = get_average(mlp_row)
            # mid_avg = get_average(mid_mlp_row)
            # , mlp_row+[mlp_avg], mid_mlp_row+[mid_avg]
            sum_rows = [head_row+[head_avg],layer_row+[layer_avg]]
            writer.writerows(sum_rows)
            print(f"save done!  {file_name}")
    
        