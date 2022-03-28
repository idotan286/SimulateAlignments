from pipeline_click import pipeline
import sys
import os
import random
from ete3 import Tree


flag = sys.argv[1]
path = sys.argv[2]
min_seq_length = int(sys.argv[3])
max_seq_length = int(sys.argv[4])
min_brach_length = float(sys.argv[5])
max_brach_length = float(sys.argv[6])

num_of_samples = int(sys.argv[7])
num_of_samples_per_file = int(1e5)

input_minIR = float(sys.argv[8])
input_maxIR = float(sys.argv[9])

input_minAVal = float(sys.argv[10])
input_maxAVal = float(sys.argv[11])

pipeline_path = f"/a/home/cc/tree/taucc/students/lifesci/edodotan/projects/deeplearning_transformer_alignment/DeepLearningMSA/sparta/"
res_path = f"{path}res_{flag}"
data_set_path = f"{path}data_set_{flag}"

if not os.path.isdir(res_path):
	os.mkdir(res_path)

if not os.path.isdir(data_set_path):
	os.mkdir(data_set_path)


msa_filename = "temp_msa.fasta"
tree_filename = "temp_tree.tree"

msa_path = os.path.join(res_path, msa_filename)
tree_path = os.path.join(res_path, tree_filename)

for sample in range(num_of_samples):
	branch_length_1 = random.uniform(min_brach_length, max_brach_length)
	branch_length_2 = random.uniform(min_brach_length, max_brach_length)
	branch_length_3 = random.uniform(min_brach_length, max_brach_length)
	branch_length_4 = random.uniform(min_brach_length, max_brach_length)
	branch_length_5 = random.uniform(min_brach_length, max_brach_length)
	branch_length_6 = random.uniform(min_brach_length, max_brach_length)
	branch_length_7 = random.uniform(min_brach_length, max_brach_length)
	branch_length_8 = random.uniform(min_brach_length, max_brach_length)
	branch_length_9 = random.uniform(min_brach_length, max_brach_length)
	branch_length_10 = random.uniform(min_brach_length, max_brach_length)
	branch_length_11 = random.uniform(min_brach_length, max_brach_length)
	branch_length_12 = random.uniform(min_brach_length, max_brach_length)
	branch_length_13 = random.uniform(min_brach_length, max_brach_length)
	branch_length_14 = random.uniform(min_brach_length, max_brach_length)
	branch_length_15 = random.uniform(min_brach_length, max_brach_length)
	branch_length_16 = random.uniform(min_brach_length, max_brach_length)
	branch_length_17 = random.uniform(min_brach_length, max_brach_length)
	branch_length_18 = random.uniform(min_brach_length, max_brach_length)
	seq_length = random.randint(min_seq_length, max_seq_length)
	t = Tree()
	t.populate(10, ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
	tree = t.write(format=1)
	tree = tree.replace(':1', f':{branch_length_1}', 1)
	tree = tree.replace(':1', f':{branch_length_2}', 1)
	tree = tree.replace(':1', f':{branch_length_3}', 1)
	tree = tree.replace(':1', f':{branch_length_4}', 1)
	tree = tree.replace(':1', f':{branch_length_5}', 1)
	tree = tree.replace(':1', f':{branch_length_6}', 1)
	tree = tree.replace(':1', f':{branch_length_7}', 1)
	tree = tree.replace(':1', f':{branch_length_8}', 1)
	tree = tree.replace(':1', f':{branch_length_9}', 1)
	tree = tree.replace(':1', f':{branch_length_10}', 1)
	tree = tree.replace(':1', f':{branch_length_11}', 1)
	tree = tree.replace(':1', f':{branch_length_12}', 1)
	tree = tree.replace(':1', f':{branch_length_13}', 1)
	tree = tree.replace(':1', f':{branch_length_14}', 1)
	tree = tree.replace(':1', f':{branch_length_15}', 1)
	tree = tree.replace(':1', f':{branch_length_16}', 1)
	tree = tree.replace(':1', f':{branch_length_17}', 1)
	tree = tree.replace(':1', f':{branch_length_18}', 1)

	with open(tree_path,'w') as f:
		f.write(tree)

	with open(msa_path,'w') as f:
		f.write(">A\n")
		f.write("T"*seq_length + "\n")
		f.write(">B\n")
		f.write("T"*seq_length + "\n")
		f.write(">C\n")
		f.write("T"*seq_length + "\n")
		f.write(">D\n")
		f.write("T"*seq_length + "\n")
		f.write(">E\n")
		f.write("T"*seq_length + "\n")
		f.write(">F\n")
		f.write("T"*seq_length + "\n")
		f.write(">G\n")
		f.write("T"*seq_length + "\n")
		f.write(">H\n")
		f.write("T"*seq_length + "\n")
		f.write(">I\n")
		f.write("T"*seq_length + "\n")
		f.write(">J\n")
		f.write("T"*seq_length + "\n")

	skip_config = {
		"sparta": True,
		"mafft": True,
		"inference": True ,
		"correct_bias": True
	}
 
	submodel_params_ = {
		"mode": "amino",
    #"mode": "nuc",
		#"submodel": "GTR",
		#"freq": (0.369764, 0.165546, 0.306709, 0.157981),
		#"rates": (0.443757853, 0.084329474, 0.115502265, 0.107429571, 0.000270340),
		#"inv_prop": 0.0,
		#"gamma_shape": 99.852225,
		#"gamma_cats": 4
	}

	res_dir = res_path #results path
	clean_run = False
	op_sys= 'linux'
	msa_filename = "temp_msa.fasta"
	tree_filename = "temp_tree.tree"
	minIR=input_minIR
	maxIR=input_maxIR
	minAVal=input_minAVal
	maxAVal=input_maxAVal
	verbose = 0
	b_num_top=100
	num_alignments = 1
	filter_p = (0.9,15)
	num_simulations = 1
	num_burnin = 1
	
	if sample % num_of_samples_per_file == 0:
		file = os.path.join(data_set_path,f'all_data_{sample}_{sample + num_of_samples_per_file}')
	with open(file, 'a') as f:
		msa = pipeline(skip_config=skip_config,
			pipeline_path=pipeline_path,
			res_dir=res_dir, 
			clean_run=clean_run,
			minAVal=minAVal,
			maxAVal=maxAVal,
			msa_filename=msa_filename,
			tree_filename=tree_filename,
			minIR=minIR,
			maxIR=maxIR,
			op_sys=op_sys,
			verbose=verbose,
			b_num_top=b_num_top,
			num_alignments=num_alignments,
			filter_p=filter_p, 
			submodel_params=submodel_params_,
			num_simulations=num_simulations,
			num_burnin=num_burnin
		)
		params_file = os.path.join(res_path, 'SpartaABC_data_name_iddif.posterior_params')
		with open(params_file, 'r') as f_params:
			lines = f_params.readlines()
			#print(f'lines\n {lines[4]}')
			summary_stat = lines[4][:-1].split("\t")[1:]
			#print(f'summary_stat\n {summary_stat}')
			f.write(",".join(summary_stat) + ',')
		seqs = msa.split("\n")
		f.write(tree + "\n")
		f.write(seqs[1] + "\n")
		f.write(seqs[3] + "\n")
		f.write(seqs[5] + "\n")
		f.write(seqs[7] + "\n")
		f.write(seqs[9] + "\n")
		f.write(seqs[11] + "\n")
		f.write(seqs[13] + "\n")
		f.write(seqs[15] + "\n")
		f.write(seqs[17] + "\n")
		f.write(seqs[19] + "\n")
		#print(f'seqs\n {seqs}')
	