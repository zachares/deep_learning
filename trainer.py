import sys
import torch
import torch.optim as optim
import torch.autograd as autograd
import utils_sl as sl
from models_modules import ModelWrapper

from typing import Dict
from types import FunctionType

# NOTE: this model only supports ADAM for optimization
class Trainer(object):
    """ A class which trains models composed of neural network and also
        calculates the evaluation metrics for the training

        Attributes:
            train_dict: a dictionary containing the hyperparameters to use
            for the optimizer that will update the weights of the models

            model_dict: a dictionary of the models that could be trained

            loss_dict: a dictionary of the loss functions that could be
            used to train the models

            eval_dict: a dictionary of the evaluation metric that could 
            be used to evaluate training

            log_dict: a dictionary to store the calculated evaluation metrics
            and loss values of training

            info_flow : a dictionary containing the input and output
            arguments to use calculating the losses and evaluation metrics
            specified for this training run

            device: a reference to the hardware the models will be trained
            on (GPU or CPU)

            opt_dict : a dictionary with the optimizer used to train each
            model

            model_inputs : a dictionary where the per batch inputs for
            the model are stored

            model_outputs : a dictionary where the per batch model
            outputs are stored
    """
    def __init__(self,
                 train_dict : Dict[str, float],
                 model_dict : Dict[str, ModelWrapper],
                 loss_dict : Dict[str, FunctionType],
                 eval_dict : Dict[str, FunctionType],
                 log_dict : dict,
                 info_flow : dict,
                 device : torch.device):
        """ Inits a Trainer instance """
        self.model_dict = model_dict	
        self.loss_dict = loss_dict
        self.eval_dict = eval_dict
        self.log_dict = log_dict
        self.info_flow = info_flow
        self.device = device

        # Setting up optimizer for models
        self.opt_dict = {}
        for key in self.model_dict.keys():
            if self.info_flow[key]["train"] == 1:
                print("\nTraining " , key, "\n")
                parameters_list = list(self.model_dict[key].parameters())
                self.opt_dict[key] = optim.Adam(parameters_list,
                                                lr=train_dict['lrn_rate'],
                                                betas=(train_dict['beta1'], train_dict['beta2']),
                                                weight_decay = train_dict['regularization_weight'])
            else:
                print("\nNot Training ", key, "\n")

    def train(self, sample_batched : Dict[str, torch.Tensor]):
        """ Performs a training step for the models in the instances
            model dict which includes a forward pass through the network
            and a update to the model weights based on the gradient of
            the models' loss functions

            Args:
                sample_batched : a dictionary containing a torch.Tensors
                which are the input arguments for the models during the
                forward pass
        """
        for key in self.model_dict.keys():
            # checking to see if the info_flow attribute specifies to 
            # train the model
            if self.info_flow[key]["train"] == 1:
                self.model_dict[key].train()

                # performing a forward pass where the gradients of the pass
                # are calculated
                with torch.enable_grad():
                    loss = self.forward_pass(sample_batched)

                for key in self.opt_dict.keys():
                    self.opt_dict[key].zero_grad()
                
                # calculating gradient
                loss.backward()

                # updating model weights
                for key in self.opt_dict.keys():
                    self.opt_dict[key].step()
            else:
                self.model_dict[key].eval()
                
                # performing a forward pass where the gradients of the pass
                # are ignored
                with torch.no_grad():
                    loss = self.forward_pass(sample_batched)

    def eval(self, sample_batched : Dict[str, torch.Tensor]):
        """ Performs an  for the models in the instances
            model dict which includes a forward pass through the network

            Args:
                sample_batched : a dictionary containing a torch.Tensors
                which are the input arguments for the models during the
                forward pass
        """

        # performing a forward pass where the gradients of the pass
        # are ignored
        with torch.no_grad():
            for key in self.model_dict.keys():
                self.model_dict[key].eval()

            loss = self.forward_pass(sample_batched)

    def forward_pass(self, sample_batched  : Dict[str, torch.Tensor]):
        """ A forward pass through the network, this method collects
            the inputs for each model in the attribute model_inputs
            and stores the outputs of each model in the attribute
            model_outputs. Then the network calculates the loss values
            and evaluations metric values for the batch.

            Args:
                sample_batched : a dictionary containing a torch.Tensors
                which are the input arguments for the models during the
                forward pass
            
            Raises:
                LookupError: if a required model input is not available in
                model_outputs
        """
        # Clearing previous results, if they exist by creating a new
        # dictionary
        self.model_inputs = {}
        self.model_outputs = {}
        self.model_outputs['dataset'] = {}

        # adding inputs from data set to model_outputs dict
        for key in sample_batched.keys():
            self.model_outputs['dataset'][key] = sample_batched[key].to(self.device)

        for key in self.model_dict.keys():
            self.model_inputs[key] = {}
            
            # collecting required inputs for model
            for input_key in self.info_flow[key]['inputs'].keys():
                input_source = self.info_flow[key]['inputs'][input_key]

                if input_key in self.model_outputs[input_source].keys():
                    self.model_inputs[key][input_key] = self.model_outputs[input_source][input_key]
                else:
                    raise LookupError("Required Model Input {} has not been calculated by {}".format(input_key, input_source))
            
            # performing forward pass through actual model
            self.model_outputs[key] = self.model_dict[key](self.model_inputs[key])

        # calculating loss and eval metric values
        return self.loss_and_evals()

    def loss_and_evals(self):
        """ Calculates the loss and eval metric values for the models that
            are being trained based on the batch on inputted data
        """
        # Creating default loss tensor to add the loss values for training
        # each model to
        loss = torch.zeros(1).float().to(self.device)

        for model_key in self.model_dict.keys():
            # Calculating and storing loss values
            for loss_name in self.info_flow[model_key]['losses'].keys():
                
                loss_args = self.info_flow[model_key]['losses'][loss_name]

                input_list = []
                for i_name, i_source in loss_args['inputs'].items():
                    input_list.append(self.model_outputs[i_source][i_name])

                log_name = loss_args['logging_name']                   
                loss_function = self.loss_dict[loss_name]
                loss_weight = loss_args['weight']

                loss += loss_function.forward(tuple(input_list),
                                              loss_weight,
                                              self.log_dict,
                                              "{}/{}".format(model_key, log_name))

            # Calculating and storing eval values
            if 'evals' in self.info_flow[model_key].keys():
                for eval_name  in self.info_flow[model_key]['evals'].keys():
                    eval_args = self.info_flow[model_key]['evals'][eval_name]

                    input_list = []
                    for i_name, i_source in eval_args['inputs'].items():
                        input_list.append(self.model_outputs[i_source][i_name])

                    log_name = eval_args['logging_name']                   
                    eval_function = self.eval_dict[eval_name]

                    eval_function.measure(tuple(input_list),
                                          self.log_dict,
                                          "{}/{}".format(model_key, log_name))
        return loss

    def save(self, epoch_num : int, model_dir : str):
        """ Saves the weights of the models in the Trainer instance's 
            model_dict attribute to a directory

            NOTE: this method saves the models both as a pkl and as
            as a set of files just with the weights of the models

            Args:
                epoch_num : the number of training epochs that have passed
                model_dir : the directory to save the model in
        """
        #saving each model_module's state dictionary and model dictionary
        for key, model in self.model_dict.items():
            model_dict = {
                key: model,
            }
            torch.save(model_dict,
                       "{}{}_{}.pkl".format(model_dir, key, str(epoch_num).zfill(6)))

            print("Saving model {} to {}{}_{}.pkl".format(key, model_dir, key, str(epoch_num).zfill(6)))
            model.save(epoch_num, model_dir)	

### OLD CODE 
# if 'outputs' in self.info_flow[model_key].keys():
#     for output_key in self.info_flow[model_key]['outputs'].keys():
#         input_list = [self.model_outputs[model_key][output_key]]

#         if 'inputs' in list(self.info_flow[model_key]['outputs'][output_key].keys()):
#             for input_key in self.info_flow[model_key]['outputs'][output_key]['inputs'].keys():
#                 input_source = self.info_flow[model_key]['outputs'][output_key]['inputs'][input_key]
#                 if input_source == "":
#                     input_source = model_key
                    
#                 input_list.append(self.model_outputs[input_source][input_key])

        # if 'losses' in list(self.info_flow[model_key]['outputs'][output_key].keys()):

        #     for loss_function_name in list(self.info_flow[model_key]['outputs'][output_key]["losses"].keys()):
        #         loss_function = self.loss_dict[loss_function_name]

        #         loss_dict = self.info_flow[model_key]['outputs'][output_key]["losses"][loss_function_name] 
        #         logging_name = loss_dict["logging_name"]

        #         loss += loss_function.forward(tuple(input_list), self.log_dict, model_key + "/" + logging_name, loss_dict)

        # if "evals" in list(self.info_flow[model_key]["outputs"][output_key].keys()):
            
        #     for metric_name in list(self.info_flow[model_key]['outputs'][output_key]["evals"].keys()):
        #         eval_function = self.eval_dict[metric_name] 

        #         eval_dict = self.info_flow[model_key]['outputs'][output_key]["evals"][metric_name]
        #         logging_name = eval_dict["logging_name"]

        #         eval_function.measure(tuple(input_list), self.log_dict, model_key + "/" + logging_name, eval_dict)



