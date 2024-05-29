import random
import glob
import numpy as np
import sys
import os
import subprocess
import time
import sys
import pandas as pd
from fairseq.models.transformer import TransformerModel
import json
from itertools import permutations
import random

#### correct
FILE_2_TEST = sys.argv[1]
RESULTS_PATH = sys.argv[2]
NUMBER_OF_INPUT_SEQUENCES = int(sys.argv[3])
TRANSFORMER_PATH_1 = sys.argv[4] 
CHECKPOINT_TRANSFORMER_1 = sys.argv[5] 
TRANSFORMER_PATH_2 = sys.argv[6]
CHECKPOINT_TRANSFORMER_2 = sys.argv[7] 
flag = int(sys.argv[8])
NUM_PROCESS = int(sys.argv[9])
NUM_PERMUTATION_PER_TRANSFOMRER = int(sys.argv[10])
TOKENZIER_PATH = sys.argv[11]

tmp_running_path = os.path.join(RESULTS_PATH, f'running_files')
if not os.path.exists(tmp_running_path):
    os.makedirs(tmp_running_path)
tmp_running_path = os.path.join(tmp_running_path, f'permutations_{NUM_PERMUTATION_PER_TRANSFOMRER}')
# create folder
if not os.path.exists(tmp_running_path):
    os.makedirs(tmp_running_path)

path_2_save_csv = f'{tmp_running_path}/certinty_and_scores_{flag}.csv'

if os.path.exists(path_2_save_csv):
    exit(f'path_2_save_csv = {path_2_save_csv} already exists')
    
transformer_folders = [(TRANSFORMER_PATH_1, CHECKPOINT_TRANSFORMER_1)]
if TRANSFORMER_PATH_2 != '':
    pair_2 = (TRANSFORMER_PATH_2, CHECKPOINT_TRANSFORMER_2)
    transformer_folders.append(pair_2)

print(f'starting script data_set_path {FILE_2_TEST} NUMBER_OF_INPUT_SEQUENCES {NUMBER_OF_INPUT_SEQUENCES}')

def calc_char_position(np_row, index):
    char = np_row[index]
    gaps_amount = np.count_nonzero(np_row == '-')
    if char == '-':
        return '-'
    return index - gaps_amount

def calc_score(true_alignment, inferred_alignment):
    cols_true = true_alignment.shape[1]
    true_tuples_list = []
    score = 0
    for i in range(0, cols_true):
        pos_true_list = []
        for j in range(NUMBER_OF_INPUT_SEQUENCES):
            pos_true_list.append(calc_char_position(true_alignment[j, :i + 1], i))
        current_tuple_true = tuple(pos_true_list)
        true_tuples_list.append(current_tuple_true)
    
    cols_inf = inferred_alignment.shape[1]
    infer_tuples_list = []
    for i in range(0, cols_inf):
        pos_inferred_list = []
        for j in range(NUMBER_OF_INPUT_SEQUENCES):
            pos_inferred_list.append(calc_char_position(inferred_alignment[j, :i + 1], i))
        current_tuple = tuple(pos_inferred_list)
        infer_tuples_list.append(current_tuple)
        if current_tuple in true_tuples_list:
            score += 1
    return score / cols_true

def convert_np_2_list_of_points(alignment_np):
    tuples_list = []
    for i in range(0, alignment_np.shape[1]):
        pos_list = []
        for j in range(NUMBER_OF_INPUT_SEQUENCES):
            pos_list.append(calc_char_position(alignment_np[j, :i + 1], i))
        current_tuple = tuple(pos_list)
        tuples_list.append(current_tuple)
    return tuples_list

def create_np_from_seqs(seqs):
    np_res = np.array([list(seqs[0])])
    for i in range(1, NUMBER_OF_INPUT_SEQUENCES):
        np_res = np.append(np_res, np.array([list(seqs[i])]), axis=0)
    return np_res

def break_line_2_rows(alignment_one_liner):
    seqs = []
    for i in range(NUMBER_OF_INPUT_SEQUENCES):
        seqs.append(alignment_one_liner[i::NUMBER_OF_INPUT_SEQUENCES])
    return seqs

def creat_source_one_liner(perm, unaligned_seqs):
    ordered_list_unaligned = []
    for idx in perm:
        ordered_list_unaligned.append(unaligned_seqs[idx])
    one_liner = "|".join(ordered_list_unaligned)
    one_liner_with_spaces = " ".join(list(one_liner))
    return one_liner_with_spaces

def break_and_order_alignment_result(perm, one_liner_res):
    words = one_liner_res.split()

    seqs = [None] * NUMBER_OF_INPUT_SEQUENCES
    # appends the seqs by the index from the permuation
    for idx, number_of_alignment in enumerate(range(NUMBER_OF_INPUT_SEQUENCES)):
        seqs[perm[idx]] = ''.join(words[number_of_alignment::NUMBER_OF_INPUT_SEQUENCES])
    
    for i in range(NUMBER_OF_INPUT_SEQUENCES):
        if len(seqs[0]) != len(seqs[i]):
            return None, None
    
    np_res = create_np_from_seqs(seqs)
    return seqs, np_res

def remove_all_but_letters(seq):
    res = ''
    for char in seq:
        if char.isalpha():
            res += char
    return res

if __name__ == "__main__":
    perm = list(permutations(list(range(NUMBER_OF_INPUT_SEQUENCES))))
    permutation_2_test = random.sample(perm, NUM_PERMUTATION_PER_TRANSFOMRER)
    print(permutation_2_test)
    with open(FILE_2_TEST, 'r') as f:
        data = json.load(f)
    
    transformers = []
    for transformer_path, transformer_checkpoint in transformer_folders:
        transformers.append(TransformerModel.from_pretrained(transformer_path, data_name_or_path=TOKENZIER_PATH, checkpoint_file=transformer_checkpoint, max_len_a=6, source_lang='source', target_lang='target'))
    
    each_process_runs = int(len(data) / NUM_PROCESS)
    print(f'each_process_runs = {each_process_runs}')
    
    list_of_uncertinty_and_score_results = []
    for idx, sample in enumerate(data):
        if not flag * each_process_runs <= idx < (flag + 1) * each_process_runs:
            continue
        unaglined_and_aligned_pairs = []
        true_rows = break_line_2_rows(sample['target'])
        for idx_unalgined, unaligned_seq in enumerate(sample['ctxs']):
            pair = (unaligned_seq['text'], true_rows[idx_unalgined])
            unaglined_and_aligned_pairs.append(pair)
        print(f'###### idx = {idx} ########')
        print(unaglined_and_aligned_pairs)
        
        unaligned_seqs = [pair[0] for pair in unaglined_and_aligned_pairs]
        aligned_seqs = [pair[1] for pair in unaglined_and_aligned_pairs]
        alignment_true_np = create_np_from_seqs(aligned_seqs)
        permutations_result = []
        
        for transfomer_num, transformer in enumerate(transformers):
            for idx_permutaiton, permutaiton in enumerate(permutation_2_test):
                transformer_source = creat_source_one_liner(permutaiton, unaligned_seqs)
                #print(transformer_source)
                transformer_res = transformer.translate(transformer_source, beam=15, temperature=1.0)
                alignment_seqs, alignment_np = break_and_order_alignment_result(permutaiton, transformer_res)
                
                if alignment_seqs:
                    is_good_alignment = True
                    for i in range(NUMBER_OF_INPUT_SEQUENCES):
                        if alignment_seqs[i].replace('-','') != unaligned_seqs[i]:
                            is_good_alignment = False
                            break
                    if is_good_alignment:
                        permutations_result.append({
                            'transfomer_num': transfomer_num,
                            'idx_permutaiton': idx_permutaiton,
                            'permutaiton': permutaiton,
                            'aligned_seqs': alignment_seqs,
                            'alignment_np': alignment_np,
                            'list_of_columns': convert_np_2_list_of_points(alignment_np),
                            'score': calc_score(alignment_true_np, alignment_np)
                        })

        for res in permutations_result:
            copy_permutations_result = permutations_result[:]  # fastest way to copy
            copy_permutations_result.remove(res)
            uncertintey_dict = {
                'idx_sample': idx,
                'transfomer_num': res['transfomer_num'],
                'permutaiton': res['permutaiton'],
                'idx_permutaiton': res['idx_permutaiton'],
                'score': res['score']
            }
            list_of_columns = res['list_of_columns']
            for other_alignments in copy_permutations_result:
                list_of_other_columns = other_alignments['list_of_columns']
                for idx_col, col in enumerate(list_of_columns):
                    uncertintey_dict[f'percent_hits_col_{idx_col}'] = uncertintey_dict.get(f'percent_hits_col_{idx_col}', 0) + 1 / len(copy_permutations_result) if col in list_of_other_columns else uncertintey_dict.get(f'percent_hits_col_{idx_col}', 0)
            if 'percent_hits_col_0' in uncertintey_dict:
                uncertintey_dict['certiny_overall'] = sum([uncertintey_dict[f'percent_hits_col_{idx_col}'] for idx_col, _ in enumerate(list_of_columns)]) / len(list_of_columns)
                list_of_uncertinty_and_score_results.append(uncertintey_dict)
        
    
    df = pd.DataFrame(list_of_uncertinty_and_score_results)
    df.to_csv(path_2_save_csv, index=False)