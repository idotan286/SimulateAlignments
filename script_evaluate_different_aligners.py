import random
import glob
import numpy as np
import sys
import os
import subprocess
import time
import sys
import pandas as pd
import json
from Bio.Align.Applications import PrankCommandline, ClustalwCommandline, DialignCommandline, MuscleCommandline, TCoffeeCommandline
import random

#### correct
FILE_2_TEST = sys.argv[1]
RESULTS_PATH = sys.argv[2]
NUMBER_OF_INPUT_SEQUENCES = int(sys.argv[3])
flag = int(sys.argv[4])
NUM_PROCESS = int(sys.argv[5])

tmp_running_path = os.path.join(RESULTS_PATH, f'running_files')
if not os.path.exists(tmp_running_path):
    os.makedirs(tmp_running_path)
tmp_running_path = os.path.join(tmp_running_path, f'other_aligners')
# create folder
if not os.path.exists(tmp_running_path):
    os.makedirs(tmp_running_path)

path_2_save_csv = f'{tmp_running_path}/different_aligners_{flag}.csv'

if os.path.exists(path_2_save_csv):
    exit(f'path_2_save_csv = {path_2_save_csv} already exists')

print(f'starting script data_set_path {FILE_2_TEST} NUMBER_OF_INPUT_SEQUENCES {NUMBER_OF_INPUT_SEQUENCES}')

def calc_char_position(np_row, index):
    char = np_row[index]
    gaps_amount = np.count_nonzero(np_row == '-')
    if char == '-':
        return '-'
    return index - gaps_amount

def spaces_encoding(alignment):
    cols_num = alignment.shape[1]
    alignment_encoded = []
    for i in range(cols_num):
        MSA_column = alignment[0, i]
        for j in range(1, NUMBER_OF_INPUT_SEQUENCES):
            MSA_column += " " + alignment[j, i]
        alignment_encoded.append(MSA_column)
    return alignment_encoded

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

def translate_result(result):
    words = result.split()

    seqs = []
    for i in range(NUMBER_OF_INPUT_SEQUENCES):
        seqs.append(''.join(words[i::NUMBER_OF_INPUT_SEQUENCES]))
    
    for i in range(NUMBER_OF_INPUT_SEQUENCES):
        if len(seqs[0]) != len(seqs[i]):
            return None, None
    
    np_res = np.array([list(seqs[0])])
    for i in range(1, NUMBER_OF_INPUT_SEQUENCES):
        np_res = np.append(np_res, np.array([list(seqs[i])]), axis=0)
    return seqs, np_res

def create_fasta_input_file(path2file: str, inputs: list):
    with open(path2file, 'w') as f:
        for i in range(len(inputs)):
            f.write(f'>{i + 1}\n') #sequences name (doesn't matter)
            f.write(f'{inputs[i]}' + '\n') #sequence

def read_fasta_file_and_create_np_array(path2file: str):
    with open(path2file, 'r') as f:
        lines = f.readlines()
        seq = ''
        seqs = [''] * NUMBER_OF_INPUT_SEQUENCES
        for i, line in enumerate(lines):
            if line[:-1] == ' ':
                continue
            if line.startswith('>'):
                # when the next line starts, adding the last string
                if i != 0: #don't append the first seq
                    seqs[index] = seq
                seq = ''
                # reading the number after the '>'
                index = int(line[1:-1]) - 1
            else:
                seq += line[:-1]
            
        seqs[index] = seq #apeend last seq
        seqs = [x.upper() for x in seqs]
    
    np_res = np.array([list(seqs[0])])
    for i in range(1, NUMBER_OF_INPUT_SEQUENCES):
        np_res = np.append(np_res, np.array([list(seqs[i])]), axis=0)
    return np_res

def run_prank(sleep_time: int, flag, folder, inputs):
    sequences_file_path = f'prank.in'
    output_file_path = f'1_out{flag}'
    create_fasta_input_file(sequences_file_path, inputs)

    prank_cline = PrankCommandline(d=sequences_file_path, o=output_file_path,  # prefix only!
        f=8,  # FASTA output
        )
    
    start = time.time()
    prank_cline()
    total_time = time.time() - start
    time.sleep(sleep_time)
    return read_fasta_file_and_create_np_array(f'{output_file_path}.best.fas'), total_time

def run_T_coffee(folder, inputs):
    sequences_file_path = f'coffee.in'
    output_file_path = f'coffee.ou'
    create_fasta_input_file(sequences_file_path, inputs)
    
    clustalw_cline = TCoffeeCommandline(infile=sequences_file_path, outfile=output_file_path, output='fasta_aln')
    
    start = time.time()
    clustalw_cline()
    total_time = time.time() - start
    return read_fasta_file_and_create_np_array(output_file_path), total_time

def run_clustalw(folder, inputs):
    sequences_file_path = f'clustalw.in'
    output_file_path = f'clustalw.ou'
    create_fasta_input_file(sequences_file_path, inputs)
    
    clustalw_cline = ClustalwCommandline("clustalw2", infile=sequences_file_path, outfile=output_file_path, output='FASTA')
    
    start = time.time()
    clustalw_cline()
    total_time = time.time() - start
    return read_fasta_file_and_create_np_array(output_file_path), total_time

def run_muscle(folder, inputs):
    sequences_file_path = f'muscle.in'
    output_file_path = f'muscle.ou'
    create_fasta_input_file(sequences_file_path, inputs)
    
    muscle_cline = MuscleCommandline("muscle", input=sequences_file_path, out=output_file_path)
    
    start = time.time()
    muscle_cline()
    total_time = time.time() - start
    return read_fasta_file_and_create_np_array(output_file_path), total_time

def run_Dialign(folder, inputs):
    sequences_file_path = f'Dialign.in'
    output_file_path = f'Dialign.fa'
    create_fasta_input_file(sequences_file_path, inputs)
    
    dialign_cline = DialignCommandline(input=sequences_file_path, fn="Dialign", fa=True)
    start = time.time()
    dialign_cline()
    total_time = time.time() - start
    return read_fasta_file_and_create_np_array(output_file_path), total_time

def run_mafft(folder, inputs):
    sequences_file_path = f'mafft.in'
    output_file_path = f'mafft.ou'
    create_fasta_input_file(sequences_file_path, inputs)
    cmd = f'mafft --auto {sequences_file_path} > {output_file_path}'
    start = time.time()
    subprocess.run(cmd, shell=True)
    total_time = time.time() - start
    return read_fasta_file_and_create_np_array(output_file_path), total_time

def run_mafft_accurate(folder, inputs):
    sequences_file_path = f'mafft.in'
    output_file_path = f'mafft_accurate.ou'
    create_fasta_input_file(sequences_file_path, inputs)
    cmd = f'mafft --auto --globalpair --maxiterate 1000 {sequences_file_path} > {output_file_path}'
    start = time.time()
    subprocess.run(cmd, shell=True)
    total_time = time.time() - start
    return read_fasta_file_and_create_np_array(output_file_path), total_time

def remove_all_but_letters(seq):
    res = ''
    for char in seq:
        if char.isalpha():
            res += char
    return res

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

if __name__ == "__main__":
    with open(FILE_2_TEST, 'r') as f:
        data = json.load(f)
    
    each_process_runs = int(len(data) / NUM_PROCESS)
    print(f'each_process_runs = {each_process_runs}')
    
    current_folder_path = os.path.join(tmp_running_path, str(flag))
    if not os.path.exists(current_folder_path):
        os.makedirs(current_folder_path)
    os.chdir(current_folder_path)
    list_of_scores = []
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
        
        unaglined_seqs = [pair[0] for pair in unaglined_and_aligned_pairs]
        aligned_seqs = [pair[1] for pair in unaglined_and_aligned_pairs]
        np_res_true = create_np_from_seqs(aligned_seqs)
        
        inferred_alignments_mafft, time_mafft = run_mafft(current_folder_path, unaglined_seqs)
        score_mafft = calc_score(np_res_true, inferred_alignments_mafft)
        
        inferred_alignments_muscle, time_muscle = run_muscle(current_folder_path, unaglined_seqs)
        score_muscle = calc_score(np_res_true, inferred_alignments_muscle)
        
        inferred_alignments_clustalw, time_clustalw = run_clustalw(current_folder_path, unaglined_seqs)
        score_clustalw = calc_score(np_res_true, inferred_alignments_clustalw)
        
        inferred_alignments_Dialign, time_dialign = run_Dialign(current_folder_path, unaglined_seqs)
        score_Dialign = calc_score(np_res_true, inferred_alignments_Dialign)
        
        try:
            inferred_alignments_prank, time_prank = run_prank(1.9, flag, current_folder_path, unaglined_seqs)
        except Exception:
            inferred_alignments_prank, time_prank = run_prank(4.7, flag, current_folder_path, unaglined_seqs)
        score_prank = calc_score(np_res_true, inferred_alignments_prank)
        
        list_of_scores.append({
            'idx_sample': idx,
            'score_mafft': score_mafft,
            'score_muscle': score_muscle,
            'score_clustalw': score_clustalw,
            'score_Dialign': score_Dialign,
            'score_prank': score_prank
        })
    
    df = pd.DataFrame(list_of_scores)
    df.to_csv(path_2_save_csv, index=False)