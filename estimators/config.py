import numpy as np
import argparse
import os

class conf(object):
	def __init__(self,
				data_path = '../../processed_acdc_dataset/hdf5_files',
				output_dir = '../../models/ACDC',
				run_name = 'FCRD_ACDC_K_16',
				batch_size = 16,
				num_class = 4,
				num_channels = 1,
				num_epochs = 250,
				learning_rate = 1e-4,
				prediction_batch_size = 16,
				load_model_from = None,
				# load_model_from = os.path.join('../../models/ACDC', 'FCRD_ACDC_128', 'models','latest.ckpt'),
				resume_training = False
				):

		self.data_path = data_path
		self.output_dir = output_dir
		self.run_name = run_name
		self.batch_size = batch_size
		self.num_channels = num_channels
		self.num_class = num_class
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate
		self.prediction_batch_size = prediction_batch_size
		self.load_model_from = load_model_from
		self.resume_training = resume_training
		run_dir = os.path.join(output_dir,run_name)
		# if self.load_model_from is None:
		# 	self.load_model_from = os.path.join(run_dir,'models','latest.ckpt')
		self.summary_dir = os.path.join(output_dir, run_name,'summary')
		self.freq_list = ['per_step', 'per_100_steps', 'per_epoch']

		# Data Augmentation Parameters
		# Set patch extraction parameters
		size0 = (64, 64)
		size1 = (128, 128)
		size2 = (256, 256)
		patch_size = size1
		mm_patch_size = size1
		max_size = (256, 256)
		train_transformation_params = {
		    'patch_size': patch_size,
		    'mm_patch_size': mm_patch_size,
		    'add_noise': ['gauss', 'none1', 'none2'],
		    'rotation_range': (-5, 5),
		    'translation_range_x': (-5, 5),
		    'translation_range_y': (-5, 5),
		    'zoom_range': (0.8, 1.2),
		    # 'do_flip': (True, True),
		    }

		valid_transformation_params = {
		    'patch_size': patch_size,
		    'mm_patch_size': mm_patch_size}
		self.transformation_params = {'train': train_transformation_params,
									  'valid': valid_transformation_params,
									  'n_labels': num_class,
									  'data_augmentation': True,
									  'full_image': False,
									  'data_deformation': True,
									  'data_crop_pad': max_size}

		# Set Environment
		os.environ['CUDA_VISIBLE_DEVICES'] = '0'