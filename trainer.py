import sys
import torch
import torch.optim as optim
import torch.autograd as autograd

import yaml

from models import *
from supervised_learning_utils import *

import multinomial as multinomial
import gaussian as gaussian
import utils as utils
######### this model only supports ADAM for optimization at the moment
class Trainer(object):
	def __init__(self, cfg, models_folder, save_models_flag, device):

		batch_size = cfg['dataloading_params']['batch_size']
		regularization_weight = cfg['training_params']['regularization_weight']
		learning_rate = cfg['training_params']['lrn_rate']
		beta_1 = cfg['training_params']['beta1']
		beta_2 = cfg['training_params']['beta2']

		self.device = device
		self.batch_size = batch_size
		self.info_flow = cfg['info_flow']

		### Initializing Model ####
		print("Initializing Neural Network Models")
		self.model_inputs = {}
		self.model_outputs = {}
		self.model_dict = declare_models(cfg, models_folder, self.device)	

		#### Setting per step model update method ####
		# Adam is a type of stochastic gradient descent    
		self.opt_dict = {}
		parameters_list = []

		for key in self.model_dict.keys():
			if self.info_flow[key]["train"] == 1:
				print("Training " , key)
				parameters_list = list(self.model_dict[key].parameters())
				self.opt_dict[key] = optim.Adam(parameters_list, lr=learning_rate, betas=(beta_1, beta_2), weight_decay = regularization_weight)
			else:
				print("Not Training ", key)

		# 	if self.info_flow[key]["train"] == 1:
		# 		print("Training " , key)
		# 		parameters_list += list(self.model_dict[key].parameters())
		# 	else:
		# 		print("Not Training ", key)

		# self.opt_dict[key] = optim.Adam(parameters_list, lr=learning_rate, betas=(beta_1, beta_2), weight_decay = regularization_weight)	
		
		###############################################
		##### Declaring possible loss functions
		##############################################
		self.loss_dict = {}
		self.loss_dict["MSE"] = Proto_Loss(nn.MSELoss(reduction = "none"))
		self.loss_dict["L1"] = Proto_Loss(nn.L1Loss(reduction = "none"))
		self.loss_dict["Multinomial_NLL"] = Proto_Loss(nn.CrossEntropyLoss(reduction = "none"))
		self.loss_dict["Multinomial_NLL_Ensemble"] = Proto_Loss_Ensemble(nn.CrossEntropyLoss(reduction = "none"))
		self.loss_dict["Weighted_Multinomial_NLL_Ensemble"] = Proto_Loss_Ensemble(nn.CrossEntropyLoss(weight = torch.tensor([2,1]).float().to(self.device), reduction = "none"))
		self.loss_dict["Binomial_NLL"] = Proto_Loss(nn.BCEWithLogitsLoss(reduction = "none"))

		self.loss_dict["Multinomial_Entropy"] = Proto_Loss(multinomial.inputs2ent)
		self.loss_dict["Multinomial_KL"] = Proto_Loss(multinomial.inputs2KL)
		self.loss_dict["Multinomial_KL_Ensemble"] = Proto_Loss_Ensemble(multinomial.inputs2KL)

		self.loss_dict["Gaussian_NLL"] = Proto_Loss(gaussian.negative_log_likelihood)
		self.loss_dict["Gaussian_KL"] =  Proto_Loss(gaussian.divergence_KL)
		###############################################
		##### Declaring possible evaluation functions
		##############################################
		self.eval_dict = {}
		self.eval_dict["Multinomial_Accuracy"] = Proto_Metric(multinomial.inputs2acc, ["accuracy"])
		self.eval_dict["Multinomial_Entropy"] = Proto_Metric(multinomial.inputs2ent_metric, ["entropy", "correxample_entropy", "incorrexample_entropy"])
		self.eval_dict["Multinomial_Accuracy_Ensemble"] = Proto_Metric_Ensemble(multinomial.inputs2acc, ["accuracy"])
		self.eval_dict["Multinomial_Entropy_Ensemble"] = Proto_Metric_Ensemble(multinomial.inputs2ent_metric, ["entropy", "correxample_entropy", "incorrexample_entropy"])		
		self.eval_dict["Multinomial_Entropy_Spread_Ensemble"] = Proto_Metric_Ensemble(multinomial.inputs2gap, ["entropy_gap"])
		self.eval_dict["Gaussian_Error_Distrb"] = Proto_Metric(gaussian.params2error_metric, ["average_error", "covariance_error_Ratio"])
		self.eval_dict["Gaussian_Error_Samples"] = Proto_Metric(gaussian.samples2error_metric, ["average_error", "covariance_error_Ratio"])
		self.eval_dict["Continuous_Error"] = Proto_Metric(utils.continuous2error_metric, ["average_accuracy", "average_error_mag"])
		####################################
		##### Training Results Dictionary for logger #############
		##########################################
		self.logging_dict = {}
		self.logging_dict['scalar'] = {}
		self.logging_dict['image'] = {}

	def train(self, sample_batched):
		for key in self.model_dict.keys():
			if self.info_flow[key]["train"] == 1:
				self.model_dict[key].train()
				with torch.enable_grad():
					loss = self.forward_pass(sample_batched)

				for key in self.opt_dict.keys():
					self.opt_dict[key].zero_grad()

				loss.backward()

				for key in self.opt_dict.keys():
					self.opt_dict[key].step()
			else:
				self.model_dict[key].eval()
				with torch.no_grad():
					loss = self.forward_pass(sample_batched)

		return self.logging_dict

	def eval(self, sample_batched):
		with torch.no_grad():
			for key in self.model_dict.keys():
				self.model_dict[key].eval()

			loss = self.forward_pass(sample_batched)

		return self.logging_dict

	def forward_pass(self, sample_batched):
		self.model_outputs['dataset'] = {}

		for key in sample_batched.keys():
			self.model_outputs['dataset'][key] = sample_batched[key].to(self.device)

		for key in self.model_dict.keys():
			self.model_inputs[key] = {}
			
			for input_key in self.info_flow[key]["inputs"].keys():
				input_source = self.info_flow[key]["inputs"][input_key]

				if input_key in self.model_outputs[input_source].keys():
					self.model_inputs[key][input_key] = self.model_outputs[input_source][input_key]
				else:
					self.model_inputs[key][input_key] = None

			self.model_outputs[key] = self.model_dict[key](self.model_inputs[key])

		return self.loss_and_evals()

	def loss_and_evals(self):
		loss = torch.zeros(1).float().to(self.device)

		for idx_model, model_key in enumerate(self.model_dict.keys()):
			if 'outputs' in self.info_flow[model_key].keys():
				for idx_output, output_key in enumerate(self.info_flow[model_key]['outputs'].keys()):
					input_list = [self.model_outputs[model_key][output_key]]

					if 'inputs' in list(self.info_flow[model_key]['outputs'][output_key].keys()):
						for input_key in self.info_flow[model_key]['outputs'][output_key]['inputs'].keys():
							input_source = self.info_flow[model_key]['outputs'][output_key]['inputs'][input_key]
							if input_source == "":
								input_source = model_key
								
							input_list.append(self.model_outputs[input_source][input_key])

					if 'losses' in list(self.info_flow[model_key]['outputs'][output_key].keys()):

						for loss_function_name in list(self.info_flow[model_key]['outputs'][output_key]["losses"].keys()):
							loss_function = self.loss_dict[loss_function_name]

							loss_dict = self.info_flow[model_key]['outputs'][output_key]["losses"][loss_function_name] 
							logging_name = loss_dict["logging_name"]

							if 'weight' in loss_dict.keys():
								weight = loss_dict['weight']
							else:
								weight = 1.0

							loss += loss_function.forward(tuple(input_list), self.logging_dict, weight, model_key + "/" + logging_name)

					if "evals" in list(self.info_flow[model_key]["outputs"][output_key].keys()):
						
						for metric_name in list(self.info_flow[model_key]['outputs'][output_key]["evals"].keys()):
							eval_function = self.eval_dict[metric_name] 

							eval_dict = self.info_flow[model_key]['outputs'][output_key]["evals"][metric_name]
							logging_name = eval_dict["logging_name"]

							eval_function.measure(tuple(input_list), self.logging_dict, model_key + "/" + logging_name)

		return loss

	def load(self, path_dict = {}):
		for key in self.model_dict.keys():
			self.model_dict[key].load(path_dict)

	def save(self, epoch_num):
		for key in self.model_dict.keys():
			self.model_dict[key].save(epoch_num)		



