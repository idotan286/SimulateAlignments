from sys import maxsize
import configuration as config
import os
import subprocess




def generate_sparta_conf(msa_file_path, res_path):
	sparta_config = config.get_sparta_config()
	result_csv = os.path.join(res_path,"stats.csv")
	sparta_conf_file = os.path.join(res_path,"sparta.conf")


	sparta_config["_inputRealMSAFile"] = msa_file_path
	sparta_config["_inputTreeFileName"] = ""
	sparta_config["_outputGoodParamsFile"] = result_csv
	sparta_config["_only_real_stats"] = "1"
	with open(sparta_conf_file, "w") as fout:
		for key in sparta_config:
			to_write = f'{key} {sparta_config[key]}\n'
			fout.write(to_write)

	return sparta_conf_file, result_csv


def get_summary_stats(res_folder_path: str, msa: list, sparta_exec_path: str):
	msa_path = os.path.join(res_folder_path, 'msa.fasta')
	with open(msa_path,'w') as msa_file:
		for idx,seq in enumerate(msa):
			msa_file.write(f'>{idx}\n')
			msa_file.write(seq)
			msa_file.write("\n")

	sparta_conf_path, result_csv = generate_sparta_conf(msa_path, res_folder_path)

	subprocess.call(f'{sparta_exec_path} {sparta_conf_path}', shell=True)
	with open (result_csv) as result_file:
		lines = result_file.readlines()
		return lines[2].split()[6:]
	

if __name__ == "__main__":
	msa = [
		"LLLGVNIDHIATLRNARGTa-YPDPVQAAFIAEQAGAd-GITVHLREDRRHITDRDVRILRQTLD-----TRMNLEMAV-------TEEMLAIAVETKPHFCCLVPEKRQEVTTEGGLDVagQRDKMRDACKRLADAGIQVSLFID--ADEEQIKAAAEVGAPFIEIHTGCYADak----",
		"MRLGVNVDHVATVRQARRTf-EPSPVFAALIAQQAGAd-QITLHLREDRRHIQDRDLELIKELIT-----IPVNLEMAP-------TEEMREIALRVKPDRITLVPERREEITTEGGLDVvsLKEKLKEYLKPIKEAGIEVSLFIE--AQKEQIDASVEVGADAIEIHTGRYANlwn---"
	]
	stats = get_summary_stats("/home/elyawy/", msa, "/home/elyawy/development/Msc/Thesis/Working_dir/SpartaABC/SpartaABC")
	print(stats)