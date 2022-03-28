# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:02:55 2020

@author: gillo
"""

import os
import logging
import sys
import warnings

from validation import tree_validator, msa_validator
import run_sparta_abc_single_folder_pipeline as runs
import msa_bias_corrector as corrector

# set to environment variable 'DEBUG' to 1 for full list of runtime parameters.

DEBUG = True if os.environ.get("DEBUG","0") == "0" else False


pipeline_path = os.path.dirname(os.path.abspath(__file__))
pipeline_path = pipeline_path.replace('\\','/')
if pipeline_path[-1]!='/':
	pipeline_path = pipeline_path + '/'

os.chdir(pipeline_path)

#%%


def pipeline(skip_config, res_dir, clean_run,msa_filename,tree_filename,pipeline_path='/bioseq/spartaabc/pipeline/',
			 minIR=0,maxIR=0.05, minAVal=1.001, maxAVal=2.0,op_sys='linux',verbose=0, filter_p=(0.9,15),
			 b_num_top=100, num_alignments=200, submodel_params="amino", num_simulations=100000, num_burnin=10000):
	
	res_dir = os.path.join(res_dir, '')
	if os.path.isdir(res_dir + "logs/"):
		print("using existing log directory")
	else:
		os.mkdir(res_dir + "logs/")
		print("creating log directory")
	log_dir = res_dir + "logs/"
	log_id = pipeline_path.split(r"/")[-2]
	if verbose!=2:
		if not sys.warnoptions:
			warnings.simplefilter("ignore")
	logging.basicConfig(filename=log_dir+log_id+'.log',level=logging.INFO,
					format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
					datefmt='%Y-%m-%d %H:%M:%S')  
	logger = logging.getLogger(__name__)

	validator_t = tree_validator()
	validator_m = msa_validator()
	try:
		validator_t.validate_tree(res_dir, tree_filename)
		validator_m.validate_msa(res_dir, msa_filename, submodel_params["mode"])
	except Exception as message:
		print(message)
		return


	if skip_config["sparta"]:
		runs.create_sims_from_data(data_name='', ow_flag=False,
							verbose=verbose, res_dir=res_dir,
							data_dir=log_dir,
							msa_filename=msa_filename,
							tree_filename=tree_filename,
 							minAVal = minAVal, maxAVal = maxAVal,
							minIR=minIR,maxIR=maxIR, num_alignments=num_alignments,
							cwd=pipeline_path,
							op_sys=op_sys,
							num_simulations=num_simulations,
							num_burnin=num_burnin) # Running spartaABC C++ code to get simulations (through another python script)
		if clean_run:
			os.remove(f'{res_dir}_eq.conf')
			os.remove(f'{res_dir}_dif.conf')
	else:
		logger.info("Skipping Sparta.")
		#check if sparta params file exists
		for model in ["eq", "dif"]:
			if os.path.isfile(res_dir + f'SpartaABC_data_name_id{model}.posterior_params'):
				print("retrieved existing params files.")
				logger.info("retrieved existing params files.")
			else:
				print("Could not find SpartaABC params file.")
				logger.error("Could not find SpartaABC params file.\nPlease provide the params files or run the full pipeline")
				return

	if skip_config["correct_bias"]:
		return corrector.apply_correction(skip_config=skip_config,res_path=res_dir, clean_run=clean_run,
								pipeline_path=pipeline_path,
								tree_file=tree_filename,filter_p=filter_p, submodel_params=submodel_params)
	else:
		logger.info("Skipping msa bias correction.")
		#check if sparta params file exists
		for model in ["eq", "dif"]:
			if os.path.isfile(res_dir + f'SpartaABC_msa_corrected_id{model}.posterior_params'):
				print("retrieved corrected param files.")
				logger.info("retrieved corrected param files.")
			else:
				print("Could not find corrected params file.")
				logger.error("Could not find corrected params file.\nPlease provide the param files or run without the --skip-bc option")
				return	
	return