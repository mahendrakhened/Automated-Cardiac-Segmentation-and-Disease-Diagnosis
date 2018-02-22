"""
	This code is for extraction of cardiac features from ACDC training database segmentations
"""
import os, re
import numpy as np
import pandas as pd
import sys
# print sys.path
sys.path.append("../") 
# Custom
from utils_heart import * 

HEADER = ["Name", "ED[vol(LV)]", "ES[vol(LV)]", "ED[vol(RV)]", "ES[vol(RV)]",
          "ED[mass(MYO)]", "ES[vol(MYO)]", "EF(LV)", "EF(RV)", "ED[vol(LV)/vol(RV)]", "ES[vol(LV)/vol(RV)]", "ED[mass(MYO)/vol(LV)]", "ES[vol(MYO)/vol(LV)]",
          "ES[max(mean(MWT|SA)|LA)]", "ES[stdev(mean(MWT|SA)|LA)]", "ES[mean(stdev(MWT|SA)|LA)]", "ES[stdev(stdev(MWT|SA)|LA)]", 
          "ED[max(mean(MWT|SA)|LA)]", "ED[stdev(mean(MWT|SA)|LA)]", "ED[mean(stdev(MWT|SA)|LA)]", "ED[stdev(stdev(MWT|SA)|LA)]", "GROUP"]

def calculate_metrics_for_training(data_path_list, name='train'):

	res = []
	for data_path in data_path_list:
		patient_folder_list = os.listdir(data_path)
		# print (patient_folder_list)
		for patient in patient_folder_list:
			print patient
			patient_folder_path = os.path.join(data_path, patient)
			config_file_path = os.path.join(patient_folder_path, 'Info.cfg')
			patient_data = {}
			with open(config_file_path) as f_in:
				for line in f_in:
					l = line.rstrip().split(": ")
					patient_data[l[0]] = l[1]

			# Read patient Number
			m = re.match("patient(\d{3})", patient)
			patient_No = int(m.group(1))
			# Read Diastole frame Number
			ED_frame_No = int(patient_data['ED'])
			ed_img = "patient%03d_frame%02d_gt.nii.gz" %(patient_No, ED_frame_No)
			# Read Systole frame Number
			ES_frame_No = int(patient_data['ES'])
			es_img = "patient%03d_frame%02d_gt.nii.gz" %(patient_No, ES_frame_No)

			pid = 'patient{:03d}'.format(patient_No)
			# Load data
			ed_data = nib.load(os.path.join(data_path, pid, ed_img))
			es_data = nib.load(os.path.join(data_path, pid, es_img))

			ed_lv, ed_rv, ed_myo = heart_metrics(ed_data.get_data(),
			                ed_data.header.get_zooms())
			es_lv, es_rv, es_myo = heart_metrics(es_data.get_data(),
			                es_data.header.get_zooms())
			ef_lv = ejection_fraction(ed_lv, es_lv)
			ef_rv = ejection_fraction(ed_rv, es_rv)

			myo_properties = myocardial_thickness(os.path.join(data_path, pid, es_img))
			es_myo_thickness_max_avg = np.amax(myo_properties[0])
			es_myo_thickness_std_avg = np.std(myo_properties[0])
			es_myo_thickness_mean_std = np.mean(myo_properties[1])
			es_myo_thickness_std_std = np.std(myo_properties[1])

			myo_properties = myocardial_thickness(os.path.join(data_path, pid, ed_img))
			ed_myo_thickness_max_avg = np.amax(myo_properties[0])
			ed_myo_thickness_std_avg = np.std(myo_properties[0])
			ed_myo_thickness_mean_std = np.mean(myo_properties[1])
			ed_myo_thickness_std_std = np.std(myo_properties[1])
			# print (es_myo_thickness_max_avg, es_myo_thickness_std_avg, es_myo_thickness_mean_std, es_myo_thickness_std_std,
			#      ed_myo_thickness_max_avg, ed_myo_thickness_std_avg, ed_myo_thickness_std_std, ed_myo_thickness_std_std)



			heart_param = {'EDV_LV': ed_lv, 'EDV_RV': ed_rv, 'ESV_LV': es_lv, 'ESV_RV': es_rv,
			       'ED_MYO': ed_myo, 'ES_MYO': es_myo, 'EF_LV': ef_lv, 'EF_RV': ef_rv,
			       'ES_MYO_MAX_AVG_T': es_myo_thickness_max_avg, 'ES_MYO_STD_AVG_T': es_myo_thickness_std_avg, 'ES_MYO_AVG_STD_T': es_myo_thickness_mean_std, 'ES_MYO_STD_STD_T': es_myo_thickness_std_std,
			       'ED_MYO_MAX_AVG_T': ed_myo_thickness_max_avg, 'ED_MYO_STD_AVG_T': ed_myo_thickness_std_avg, 'ED_MYO_AVG_STD_T': ed_myo_thickness_mean_std, 'ED_MYO_STD_STD_T': ed_myo_thickness_std_std,}
			r=[]

			r.append(pid)
			r.append(heart_param['EDV_LV'])
			r.append(heart_param['ESV_LV'])
			r.append(heart_param['EDV_RV'])
			r.append(heart_param['ESV_RV'])
			r.append(heart_param['ED_MYO'])
			r.append(heart_param['ES_MYO'])
			r.append(heart_param['EF_LV'])
			r.append(heart_param['EF_RV'])
			r.append(ed_lv/ed_rv)
			r.append(es_lv/es_rv)
			r.append(ed_myo/ed_lv)
			r.append(es_myo/es_lv)
			# r.append(patient_data[pid]['Height'])
			# r.append(patient_data[pid]['Weight'])
			r.append(heart_param['ES_MYO_MAX_AVG_T'])
			r.append(heart_param['ES_MYO_STD_AVG_T'])
			r.append(heart_param['ES_MYO_AVG_STD_T'])
			r.append(heart_param['ES_MYO_STD_STD_T'])

			r.append(heart_param['ED_MYO_MAX_AVG_T'])
			r.append(heart_param['ED_MYO_STD_AVG_T'])
			r.append(heart_param['ED_MYO_AVG_STD_T'])
			r.append(heart_param['ED_MYO_STD_STD_T'])
			r.append(patient_data['Group'])
			res.append(r)

		df = pd.DataFrame(res, columns=HEADER)
		if not os.path.exists('./training_data'):
			os.makedirs('./training_data')	
		df.to_csv("./training_data/Cardiac_parameters_{}.csv".format(name), index=False)

if __name__ == '__main__':
	#Path to ACDC training set
	train_path = ['../../processed_acdc_dataset/dataset/train_set']
	validation_path = ['../../processed_acdc_dataset/dataset/validation_set', '../../processed_acdc_dataset/dataset/test_set']
	full_train = ['../../processed_acdc_dataset/dataset/train_set', '../../processed_acdc_dataset/dataset/validation_set', 
				  '../../processed_acdc_dataset/dataset/test_set']
	calculate_metrics_for_training(train_path, name='train')
	calculate_metrics_for_training(validation_path, name='validation')
	calculate_metrics_for_training(full_train, name='training')