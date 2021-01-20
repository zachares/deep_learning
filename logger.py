import os
import sys
import time
import datetime
import random
import numpy as np
import torch
import yaml
import pickle

from tensorboardX import SummaryWriter

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt 

class Logger(object):
	def __init__(self, cfg, debugging_flag, save_model_flag, run_description):
		self.debugging_flag = debugging_flag
 
		trial_description = cfg['logging_params']['run_notes']
		logging_folder = cfg['logging_params']['logging_folder']

		##### Code to keep track of runs during a day and create a unique path for logging each run
		with open("run_tracking.yml", 'r+') as ymlfile1:
			load_cfg = yaml.safe_load(ymlfile1)

		t_now = time.time()

		date = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d')

		# print("Run tracker Date: ", load_cfg['run_tracker']['date'] ,"  Actual Date: ", date)
		if load_cfg['run_tracker']['date'] == date:
			load_cfg['run_tracker'][run_description] +=1
		else:
			print("New day of training!")
			load_cfg['run_tracker']['date'] = date
			load_cfg['run_tracker']['debugging'] = 0
			load_cfg['run_tracker']['training'] = 0        
			load_cfg['run_tracker']['testing'] = 0

		with open("run_tracking.yml", 'w') as ymlfile1:
			yaml.dump(load_cfg, ymlfile1)

		if self.debugging_flag == False or save_model_flag:
			run_tracking_num = load_cfg['run_tracker'][run_description]

			if os.path.isdir(logging_folder) == False:
				os.mkdir(logging_folder)

			self.logging_folder = logging_folder + date + "_"+ run_description +\
				 "_" + str(run_tracking_num) + trial_description + "/"

			if os.path.isdir(self.logging_folder) == False:
				os.mkdir(self.logging_folder)

			print("Logging and Model Saving Folder: ", self.logging_folder)
				
			self.writer = SummaryWriter(self.logging_folder)

	def save_scalars(self, logging_dict, iteration, label):
		if self.debugging_flag == False and len(logging_dict['scalar'].keys()) != 0:
			for key in logging_dict['scalar'].keys():
				# print(key, logging_dict['scalar'][key])
				self.writer.add_scalar(label + key, logging_dict['scalar'][key], iteration)

	def save_dict(self, name, dictionary, yml_bool, folder = None):
		if self.debugging_flag == False:
			if folder == None:
				save_folder = self.logging_folder
			else:
				save_folder = folder

			if yml_bool:
				print("Saving ", name, " to: ", save_folder + name + ".yml")

				with open(save_folder + name + ".yml", 'w') as ymlfile2:
					yaml.dump(dictionary, ymlfile2)

			else:
				print("Saving ", name, " to: ", save_folder + name + ".pkl")
				with open(save_folder + name + '.pkl', 'wb') as f:
					pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)

	def save_images2D(self, logging_dict, iteration, label):
		if self.debugging_flag == False and len(list(logging_dict['image'].keys())) != 0:
			for image_key in logging_dict['image']:

				image_list = logging_dict['image'][image_key]

				if len(image_list) != 0:
					for idx, image in enumerate(image_list):
						image_list[idx] = image.detach().cpu().numpy()

					image_array = np.rot90(np.concatenate(image_list, axis = 1), k = 3, axes=(1,2)).astype(np.uint8)

					self.writer.add_image(label + image_key + 'predicted_image' , image_array, iteration)

	def save_npimages2D(self, logging_dict, iteration, label):
		if self.debugging_flag == False and len(list(logging_dict['image'].keys())) != 0:
			for image_key in logging_dict['image']:

				image_list = logging_dict['image'][image_key]

				if len(image_list) != 0:
					image_array = np.expand_dims(np.concatenate(image_list, axis = 1), axis = 0)

					self.writer.add_image(label + image_key + 'visitation_freq' , image_array, iteration)

	def save_tsne(self, points, labels_list, iteration, label, tensor_bool):	
		if self.debugging_flag == False:
			perplexity = 30.0
			lr_rate = 500
			initial_dims = 30
			tsne = TSNE(n_components=2, perplexity = 30.0, early_exaggeration = 12.0, learning_rate = 200.0, n_iter = 1000, method='barnes_hut')
			print("Beginning TSNE")
			if tensor_bool:
				Y = tsne.fit_transform(points.detach().cpu().numpy())
			else:
				Y = tsne.fit_transform(points)

			print("Finished TSNE")

			for idx, label_tuple in enumerate(labels_list):
				description, labels = label_tuple
				plt.switch_backend('agg')
				fig = plt.figure()
				if tensor_bool:
					plt.scatter(Y[:,0], Y[:,1], c = labels.detach().cpu().numpy())
				else:
					plt.scatter(Y[:,0], Y[:,1], c = labels)

				self.writer.add_figure(label + '_tsne_' + description, fig, iteration)