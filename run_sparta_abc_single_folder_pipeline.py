# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:43:59 2020

@author: gillo
"""

import os
import subprocess
import shutil
import logging
from  configuration import get_sparta_config
logger = logging.getLogger(__name__)

from Bio import AlignIO


def phylip2fasta(data_path,dry_run_flag=False, verbose=1):
	if dry_run_flag:
		if verbose:
			print('dry run: entered phylip2fast')
			logger.info(data_path+' dry run: entered phylip2fast')
		return

	AlignIO.convert(os.path.join(data_path,'ref_msa.aa.phy'), 'phylip-relaxed', os.path.join(data_path,'ref_msa.aa.fasta'),'fasta')
	logger.info(data_path + " Done convert phylip to fasta")
	if verbose:
		print("Done convert phylip to fasta")
	return

def run_sparta_abc(exe_path,conf_path):
	args = exe_path + ' ' + conf_path
	tmp = subprocess.call(args,shell=True)
	return tmp



def create_sims(data_name,verbose=1,res_dir='results/',
				data_dir='data',msa_filename='ref_msa.aa.fasta',
				tree_filename='RAxML_tree.tree',
				minIR = 0, maxIR = 0.05, minAVal = 1.001, maxAVal = 2.0, num_alignments=200,
				cwd='/groups/pupko/gilloe/spartaABC/code/abc_nn/',
				op_sys='linux', num_simulations=100000, num_burnin=10000):
	logger.info('Inside create_sims function.')

	out_dir = res_dir + data_name

	model_types = ["dif"]#["eq", "a_dif", "r_dif", "dif"]

	logger.info('Writing conf files.')
	for model in model_types:
		sparta_config = get_sparta_config()

		sparta_config["_numSimulations"] = str(num_simulations)
		sparta_config["_numBurnIn"] = str(num_burnin)

		sparta_config['_outputGoodParamsFile'] = os.path.join(out_dir,f"SpartaABC_data_name_id{model}.posterior_params")
		sparta_config['_outputAlignmnetsFile'] = os.path.join(out_dir,f"alignments_{model}.fasta")
		sparta_config['_alignments_output'] = str(num_alignments)
		sparta_config["_inputRealMSAFile"] = os.path.join(res_dir,msa_filename)
		sparta_config["_inputTreeFileName"] = os.path.join(res_dir,tree_filename)
		sparta_config["_minIRVal"] = str(round(minIR,2))
		sparta_config["_maxIRVal"] = str(round(maxIR,2))
		sparta_config["_modelType"] = model
		sparta_config["_minAVal"] = minAVal
		sparta_config["_maxAVal"] = maxAVal

		with open(out_dir+f'_{model}.conf', "wt") as fout:
			for key in sparta_config:
				to_write = f'{key} {sparta_config[key]}\n'
				fout.write(to_write)

		logger.info(f"Wrote {out_dir}_{model}.conf. Based on {cwd}sparta_conf_template.conf")
			
	stat = "didn't run"
	if op_sys=='linux':
		logger.info(f"linux op system")
		for model in model_types:
			stat = run_sparta_abc(exe_path=cwd+'SpartaABC', conf_path=res_dir+data_name+data_name+f'_{model}.conf')
			logger.info(f"ran {cwd+f'SpartaABC'}, conf_path={res_dir+data_name+data_name+f'_{model}.conf'}, stat={stat}")
	else: #windows
		logger.info(f"windows op system")
		stat_eq = run_sparta_abc(exe_path=cwd+'SpartaABC', conf_path=res_dir+data_name+data_name+'_ideq.conf')
		logger.info(f"ran {cwd+'SpartaABC'}, conf_path={res_dir+data_name+data_name+'_ideq.conf'}")
		stat_dif = run_sparta_abc(exe_path=cwd+'SpartaABC', conf_path=res_dir+data_name+data_name+'_iddif.conf')
		logger.info(f"ran {cwd+'SpartaABC'}, conf_path={res_dir+data_name+data_name+'_iddif.conf'}")
	logger.info(f'{data_name} simulations are done - finish stats {stat}.')
	if verbose:
		print(f'{data_name} simulations are done - finish stats {stat}.')
	return stat




def create_sims_from_data(data_name,ow_flag=False,verbose=1,
						  res_dir='results',data_dir='data',
						  msa_filename='ref_msa.aa.fasta',
						  tree_filename='RAxML_tree.tree',
						  minAVal = 1.001, maxAVal = 2.0,
						  minIR = 0, maxIR = 0.05, num_alignments=200,
						  cwd='/groups/pupko/gilloe/spartaABC/code/abc_nn/',
						  op_sys='linux', num_simulations=100000, num_burnin=10000):

	if not res_dir.endswith('/'):
		res_dir += '/'
	if not data_dir.endswith('/'):
		data_dir += '/'

	logger.info(data_name +' started.')
	stat_eq = stat_dif = "didn't run"
	if verbose:
		print(data_name +' started.')
	try:
		create_sims(data_name=data_name,verbose=verbose,
					res_dir=res_dir,data_dir=data_dir,
					msa_filename=msa_filename,
					minAVal=minAVal, maxAVal=maxAVal,
					tree_filename=tree_filename,
					minIR=minIR,maxIR=maxIR, num_alignments=num_alignments,
					cwd=cwd,op_sys=op_sys, num_simulations=num_simulations, num_burnin=num_burnin)
	except:
		if ow_flag:
			shutil.rmtree(os.getcwd() + '/'+res_dir + data_name)
			create_sims(data_name=data_name,verbose=verbose,
						res_dir=res_dir,data_dir=data_dir,
						msa_filename=msa_filename,
						tree_filename=tree_filename,
						minIR=minIR,maxIR=maxIR, num_alignments=num_alignments,
						cwd=cwd,op_sys=op_sys)
		else:
			logger.info(data_name +' skipped.')
			if verbose:
				print(data_name +' skipped.')