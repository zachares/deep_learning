import torch
import torch.nn as nn
from torch.nn import functional as F 
from torch.distributions import Normal
from torch.distributions import Categorical
import torchvision
import copy
import yaml
import numpy as np

import utils_sl as sl
from typing import List, Tuple, Dict

##########################################
### Multi-layer Neural Network Modules ###
##########################################
class ModuleWrapper(nn.Module):
    """ a super class / wrapper for all multi-layer neural network 
        modules with integrated methods for loading, saving and returning
        the parameters in the module.

        Attributes:
            model_name: a string with the model name of neural network
            module
            device: a torch.device that the module performs its
            calculations on
            model: the multi-layer neural network module which performs
            the calculations
    """
    def __init__(self, model_name : str, device : torch.device):
        """ Inits a ModuleWrapper instance """
        super().__init__()
        self.model_name = model_name
        self.device = device
        # self.parallel = False

    def set_device(self, device):
        """ sets the device of the model """
        self.device = device
        self.model = self.model.to(self.device)

    def forward(self, inputs : torch.Tensor) -> torch.Tensor:
        """ Returns the output of the module given a compatible input """
        return self.model(inputs)

    def weight_parameters(self) -> List[torch.Tensor]:
        """ Returns all the weight parameters in the module """
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self) -> List[torch.Tensor]:
        """ Returns all the bias parameters in the module. This is useful
            if you want to use a different weight initialization method
            for bias terms
        """
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def save(self, epoch_num : int, model_dir : str):
        """ Saves all the weights in the module to a directory """
        ckpt_path = '{}_{}'.format(model_dir + self.model_name, str(epoch_num).zfill(6))
        print("Saved Model to: ", ckpt_path)
        torch.save(self.model.state_dict(), ckpt_path)

    def load(self, epoch_num : int, model_dir : str):
        """ Loads all the weights in the module from a directory """
        ckpt_path = '{}_{}'.format(model_dir + self.model_name, str(epoch_num).zfill(6))
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt)
        print("Loaded Model to: ", ckpt_path)

    # TODO: Debug running calculations on multiple GPUs
    # def set_parallel(self, parallel_bool):
    #     if parallel_bool:
    #        self.model =  nn.DataParallel(self.model)

    #     self.parallel = parallel_bool

class Embedding(ModuleWrapper):
    """ A ModuleWrapper Class with a model composed of an embedding layer

        Additional Attributes:
            num_embed : the number of categories that require an embedding
            vector
            embed_dim : the number of dimensions an embedding vector
            should have
        
        NOTE: Please see ModuleWrapper class for other built-in methods.
    """
    def __init__(self,
                 model_name : str,
                 num_embed : int,
                 embed_dim : int,
                 padding_idx : int = None,
                 max_norm : float = None,
                 norm_type : float = None,
                 scale_grad_by_freq : bool = False,
                 sparse : bool = False,
                 device : torch.device = None):
        """ Inits an Embedding instance 
            Please see https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
            for a description of the arguments to the init function
        """
        super().__init__(model_name + "_embed", device = device)
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.model = nn.Embedding(self.num_embed,
                                  self.embed_dim,
                                  padding_idx=padding_idx,
                                  max_norm=max_norm,
                                  norm_type=norm_type,
                                  scale_grad_by_freq=scale_grad_by_freq,
                                  sparse=sparse)

    def forward(self, idxs : torch.Tensor) -> torch.Tensor:
        """ Forward pass through the model """
        return self.model(idxs)
        
    # def set_parallel(self, bool):
    #     pass

class CONV2DN(ModuleWrapper):
    """ A ModuleWrapper Class with a model composed of a multi-layer 
        2d convolutional neural network. 

        Additional Attributes:
            input_size : a tuple of three integers describing the 
            size of the inputted images / data i.e. (inChannels x inHeight x inWidth)

            output_size : a tuple of three integers describing the size
            of the output tensor (outChannels x outHeight x outWidth)

            batchnorm: a boolean indicating whether to include batchnorm
            layers in the CNN

            nonlinear: a boolean indicating whether to include a 
            nonlinear layer i.e. ReLu after the last convolutional layer
            of the network

            droput: a boolean indicating whether to include dropout
            layers in the CNN

            dropout_prob: the dropout probability to use during training

            uc: a boolean indicating whether to keep drop out layers
            active during evalution

            layer_name_list: a list of strings with a name for each
            layer in the network module

        NOTE: This module uses a heuristic to determine the number of
        layers and the sizes of each convolutional layer required to 
        go from the requested input size to the request output size

        NOTE: Please see ModuleWrapper class for other built-in methods.
    """
    def __init__(self,
                 model_name : str,
                 input_size : Tuple[int],
                 output_size : Tuple[int],
                 nonlinear : bool = False,
                 batchnorm : bool = True,
                 dropout : bool = False,
                 dropout_prob : float = 0.5,
                 uc : bool = True,
                 device : torch.device = None):
        """ Inits a CONV2DN instance             
            Please see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
            as a reference to the convolutional layers in the network
        """
        super().__init__(model_name + "_cnn2d", device = device)

        self.input_size = input_size
        self.output_size = output_size

        self.batchnorm = batchnorm
        self.nonlinear = nonlinear
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.uc = uc

        # Calculates the number and size of each layer based on the input
        # and output size
        e_p_list = sl.get_2Dconv_params(input_size, output_size) # encoder parameters

        # print(e_p_list)
        layer_list = []
        self.layer_name_list = []

        for i, e_p in enumerate(e_p_list):
            layer_list.append(nn.Conv2d(e_p[0],
                                        e_p[1],
                                        kernel_size=(e_p[2], e_p[3]),
                                        stride=(e_p[4], e_p[5]),
                                        padding=(e_p[6], e_p[7]),
                                        bias=True))

            self.layer_name_list.append('conv2d_' + str(i))

            if i != (len(e_p_list) - 1) or nonlinear:
                if dropout:
                    layer_list.append(nn.Dropout2d(p=dropout_prob))
                    self.layer_name_list.append('dropout_' + str(i))

                if batchnorm:
                    layer_list.append(nn.BatchNorm2d(e_p[1]))
                    self.layer_name_list.append('batchnorm2d_' + str(i))

                layer_list.append(nn.LeakyReLU(0.1, inplace = False))
                self.layer_name_list.append('leaky_relu_' + str(i))

        self.model = nn.ModuleList(layer_list)

        #############################
        ### Weight Initialization ###
        #############################
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs : torch.Tensor) -> torch.Tensor:
        """ Forward pass through the model """
        for i, layer in enumerate(self.model):
            name = self.layer_name_list[i]
            if name[:4] == 'drop' and self.uc:
                layer.train()

            if i == 0:
                out = layer(inputs)
            else:
                out = layer(out)

        return out

#### a 2D deconvolutional network
class DECONV2DN(ModuleWrapper):
    """ A ModuleWrapper Class with a model composed of a multi-layer 
        2d deconvolutional neural network. 

        Additional Attributes:
            input_size : a tuple of three integers describing the 
            size of the inputted images / data i.e. (inChannels x inHeight x inWidth)

            output_size : a tuple of three integers describing the size
            of the output tensor (outChannels x outHeight x outWidth)

            batchnorm: a boolean indicating whether to include batchnorm
            layers in the deCNN

            nonlinear: a boolean indicating whether to include a 
            nonlinear layer i.e. ReLu after the last convolutional layer
            of the network

            droput: a boolean indicating whether to include dropout
            layers in the deCNN

            dropout_prob: the dropout probability to use during training

            uc: a boolean indicating whether to keep drop out layers
            active during evalution

            layer_name_list: a list of strings with a name for each
            layer in the network module

        NOTE: This module uses a heuristic to determine the number of
        layers and the sizes of each deconvolutional layer required to 
        go from the requested input size to the request output size

        NOTE: Please see ModuleWrapper class for other built-in methods.
    """
    def __init__(self,
                 model_name : str,
                 input_size : Tuple[int],
                 output_size : Tuple[int],
                 nonlinear : bool = False,
                 batchnorm : bool = True,
                 dropout : bool = False,
                 dropout_prob : float = 0.5, 
                 uc : bool = False,
                 device : torch.device = None):
        """ Inits a DECONV2DN instance             
            Please see https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d
            as a reference to the deconvolutional layers in the network
        """
        super().__init__(model_name + "_decnn2d", device = device)

        self.input_size = input_size
        self.output_size = output_size

        self.batchnorm = batchnorm
        self.nonlinear = nonlinear
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.uc = uc

        # assert self.dropout != self.batchnorm

        # Calculates the number and size of each layer based on the input
        # and output size
        e_p_list = sl.get_2Dconv_params(output_size, input_size)
        # the list must be reversed because it is a deconv layer instead
        # of a conv layer
        e_p_list.reverse()

        layer_list = []
        self.layer_name_list = []

        for i, e_p in enumerate(e_p_list):
            layer_list.append(nn.ConvTranspose2d(e_p[1],
                                                 e_p[0],
                                                 kernel_size=(e_p[2], e_p[3]),
                                                 stride=(e_p[4], e_p[5]),
                                                 padding=(e_p[6], e_p[7]),
                                                 bias=True))

            self.layer_name_list.append('deconv2d_' + str(i))

            if i != (len(e_p_list) - 1) or nonlinear:
                if dropout:
                    layer_list.append(nn.Dropout2d(p=dropout_prob))
                    self.layer_name_list.append('dropout_' + str(i))

                if batchnorm:
                    layer_list.append(nn.BatchNorm2d(e_p[0]))
                    self.layer_name_list.append('batchnorm2d_' + str(i))

                layer_list.append(nn.LeakyReLU(0.1, inplace = False))
                self.layer_name_list.append('leaky_relu_' + str(i))

        self.model = nn.ModuleList(layer_list)
        #############################
        ### Weight Initialization ###
        #############################
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs : torch.Tensor) -> torch.Tensor:
        """ Forward pass through the model """
        for i, layer in enumerate(self.model):
            name = self.layer_name_list[i]
            if name[:4] == 'drop' and self.uc:
                layer.train()

            if i == 0:
                out = layer(inputs)
            else:
                out = layer(out)

        return out

class CONV1DN(ModuleWrapper):
    """ A ModuleWrapper Class with a model composed of a multi-layer 
        1d convolutional neural network. This network can be useful
        for processing short time series (num_steps < 100) of feature
        vectors

        Additional Attributes:
            input_size : a tuple of two integers describing the 
            size of the inputted images / data i.e. (inChannels x inHeight)

            output_size : a tuple of two integers describing the size
            of the output tensor (outChannels x outHeight)

            batchnorm: a boolean indicating whether to include batchnorm
            layers in the deCNN

            nonlinear: a boolean indicating whether to include a 
            nonlinear layer i.e. ReLu after the last convolutional layer
            of the network

            droput: a boolean indicating whether to include dropout
            layers in the deCNN

            dropout_prob: the dropout probability to use during training

            uc: a boolean indicating whether to keep drop out layers
            active during evalution

            layer_name_list: a list of strings with a name for each
            layer in the network module

        NOTE: This module uses a heuristic to determine the number of
        layers and the sizes of each deconvolutional layer required to 
        go from the requested input size to the request output size

        NOTE: Please see ModuleWrapper class for other built-in methods.
    """
    def __init__(self, 
                 model_name : str,
                 input_size : Tuple[int],
                 output_size : Tuple[int],
                 nonlinear : bool = False,
                 batchnorm : bool = True,
                 dropout : bool = False,
                 dropout_prob : float = 0.5,
                 uc : bool = False,
                 device : torch.device = None):
        """ Inits a CONV1DN instance             
            Please see https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
            as a reference to the convolutional layers in the network
        """
        super().__init__(model_name + "_cnn1d", device = device)

        self.input_size = input_size
        self.output_size = output_size

        self.batchnorm = batchnorm
        self.nonlinear = nonlinear
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.uc = uc

        # assert self.dropout != self.batchnorm
        # Calculates the number and size of each layer based on the input
        # and output size
        e_p_list = sl.get_1Dconv_params(input_size, output_size)

        layer_list = []
        self.layer_name_list = []

        for i, e_p in enumerate(e_p_list):
            layer_list.append(nn.Conv1d(e_p[0],
                                        e_p[1],
                                        kernel_size= e_p[2],
                                        stride=e_p[3],
                                        padding=e_p[4],
                                        bias=True))

            self.layer_name_list.append('conv1d_' + str(i))

            if i != (len(e_p_list) - 1) or nonlinear:
                if dropout:
                    layer_list.append(nn.Dropout(p=dropout_prob))
                    self.layer_name_list.append('dropout1d_' + str(i))
                if batchnorm:
                    layer_list.append(nn.BatchNorm1d(e_p[1]))
                    self.layer_name_list.append('batchnorm1d_' + str(i))

                layer_list.append(nn.LeakyReLU(0.1, inplace = False))
                self.layer_name_list.append('leaky_relu_' + str(i))

        self.model = nn.ModuleList(layer_list)

    def forward(self, inputs : torch.Tensor) -> torch.Tensor:
        """ Forward pass through the model """
        for i, layer in enumerate(self.model):
            name = self.layer_name_list[i]
            if name[:4] == 'drop' and self.uc:
                layer.train()

            if i == 0:
                out = layer(inputs)
            else:
                out = layer(out)

        return out

class DECONV1DN(ModuleWrapper):
    """ A ModuleWrapper Class with a model composed of a multi-layer 
        1d deconvolutional neural network. This network can be useful
        for generating short time series (num_steps < 100) of feature
        vectors

        Additional Attributes:
            input_size : a tuple of two integers describing the 
            size of the inputted images / data i.e. (inChannels x inHeight)

            output_size : a tuple of two integers describing the size
            of the output tensor (outChannels x outHeight)

            batchnorm: a boolean indicating whether to include batchnorm
            layers in the deCNN

            nonlinear: a boolean indicating whether to include a 
            nonlinear layer i.e. ReLu after the last convolutional layer
            of the network

            droput: a boolean indicating whether to include dropout
            layers in the deCNN

            dropout_prob: the dropout probability to use during training

            uc: a boolean indicating whether to keep drop out layers
            active during evalution

            layer_name_list: a list of strings with a name for each
            layer in the network module

        NOTE: This module uses a heuristic to determine the number of
        layers and the sizes of each deconvolutional layer required to 
        go from the requested input size to the request output size

        NOTE: Please see ModuleWrapper class for other built-in methods.
    """
    def __init__(self,
                 model_name : str,
                 input_size : Tuple[int],
                 output_size : Tuple[int],
                 nonlinear : bool = False,
                 batchnorm : bool = True,
                 dropout : bool = False,
                 dropout_prob : float = 0.5,
                 uc : bool = False,
                 device : torch.device = None):
        super().__init__(model_name + "_decnn1d", device = device)
        """ Inits a CONV1DN instance             
            Please see https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html#torch.nn.ConvTranspose1d
            as a reference to the deconvolutional layers in the network
        """
        self.input_size = input_size
        self.output_size = output_size

        self.batchnorm = batchnorm
        self.nonlinear = nonlinear
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.uc = uc

        # assert self.dropout != self.batchnorm
        # Calculates the number and size of each layer based on the input
        # and output size
        e_p_list = sl.get_1Dconv_params(input_size, output_size)
        # the list must be reversed because it is a deconv layer instead
        # of a conv layer
        e_p_list.reverse()

        layer_list = []
        self.layer_name_list = []

        for i, e_p in enumerate(e_p_list):
            layer_list.append(nn.ConvTranspose1d(e_p[1],
                                                 e_p[0],
                                                 kernel_size= e_p[2],
                                                 stride=e_p[3],
                                                 padding=e_p[4],
                                                 bias=True))

            self.layer_name_list.append('conv1d_' + str(i))

            if i != (len(e_p_list) - 1) or nonlinear:
                if dropout:
                    layer_list.append(nn.Dropout(p=dropout_prob))
                    self.layer_name_list.append('dropout1d_' + str(i))

                if batchnorm:
                    layer_list.append(nn.BatchNorm1d(e_p[0]))
                    self.layer_name_list.append('batchnorm1d_' + str(i))

                layer_list.append(nn.LeakyReLU(0.1, inplace = False))
                self.layer_name_list.append('leaky_relu_' + str(i))

        self.model = nn.ModuleList(layer_list)

    def forward(self, inputs : torch.Tensor) -> torch.Tensor:
        """ Forward pass through the model """
        for i, layer in enumerate(self.model):
            name = self.layer_name_list[i]
            if name[:4] == 'drop' and self.uc:
                layer.train()

            if i == 0:
                out = layer(inputs)
            else:
                out = layer(out)

        return out

class FCN(ModuleWrapper):
    """ A ModuleWrapper Class with a model composed of a multi-layer 
        fully connected layers

        Additional Attributes:
            input_size: an integer describing the number of dimensions
            in an input feature vector

            output_size: an integer describing the desired number of 
            dimensions in the models output vector

            num_layers: the number of layers in the fully-connected
            network

            nonlinear: a boolean indicating whether to include a 
            nonlinear layer i.e. ReLu after the last convolutional layer
            of the network

            batchnorm: a boolean indicating whether to include batchnorm
            layers in the deCNN

            droput: a boolean indicating whether to include dropout
            layers in the deCNN

            dropout_prob: the dropout probability to use during training

            uc: a boolean indicating whether to keep drop out layers
            active during evalution

            layer_name_list: a list of strings with a name for each
            layer in the network module

        NOTE: This module uses a heuristic to determine the size of the
        intermediate layers in the network

        NOTE: Please see ModuleWrapper class for other built-in methods.
    """
    def __init__(self,
                 model_name : str,
                 input_size: int,
                 output_size : int,
                 num_layers : int,
                 nonlinear : bool = False,
                 batchnorm : bool = True,
                 dropout : bool = False,
                 dropout_prob : float = 0.5,
                 uc : bool = False,
                 device : torch.device = None):
        """ Inits a FCN instance             
            Please see https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
            as a reference to the linear layers in the network
        """
        super().__init__(model_name + "_fcn", device = device)

        self.input_size = input_size
        self.output_size = output_size

        self.num_layers = num_layers
        self.batchnorm = batchnorm
        self.nonlinear = nonlinear
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.uc = uc

        if input_size > output_size:
            middle_size = input_size
        else:
            middle_size = output_size

        # assert self.batchnorm != self.dropout

        # print("Dropout Rate: ", self.dropout_prob)
        layer_list = []
        self.layer_name_list = []

        for i in range(self.num_layers):
            if dropout:
                layer_list.append(nn.Dropout(p=dropout_prob))
                self.layer_name_list.append('dropout1d_' + str(i))

            if self.num_layers == 1:
                layer_list.append(nn.Linear(input_size, output_size))
                self.layer_name_list.append('linear_' + str(i))
            else:
                if i == (self.num_layers - 1):
                    layer_list.append(nn.Linear(middle_size, output_size))
                    self.layer_name_list.append('linear_' + str(i))
                elif i == 0:
                    layer_list.append(nn.Linear(input_size, middle_size))
                    self.layer_name_list.append('linear_' + str(i))
                else:
                    layer_list.append(nn.Linear(middle_size, middle_size))
                    self.layer_name_list.append('linear_' + str(i))


            if i != (self.num_layers - 1) or nonlinear:
                if batchnorm:
                    layer_list.append(nn.BatchNorm1d(e_p[1]))
                    self.layer_name_list.append('batchnorm1d_' + str(i))

                layer_list.append(nn.LeakyReLU(0.1, inplace = False))
                self.layer_name_list.append('leaky_relu_' + str(i))

        self.model = nn.ModuleList(layer_list)

    def forward(self, inputs : torch.Tensor) -> torch.Tensor:
        """ Forward pass through the model """
        for i, layer in enumerate(self.model):
            name = self.layer_name_list[i]
            if name[:4] == 'drop' and self.uc:
                layer.train()
            if i == 0:
                out = layer(inputs)
            else:
                out = layer(out)
        return out

class Transformer(ModuleWrapper):
    """ A ModuleWrapper Class with a model composed of transformer network

        Additional Attributes:
            input_size: an integer describing the number of dimensions
            in an input feature vector

            num_enc_layers: the number of layers in the transformer
            encoder network

            num_dec_layers: the number of layers in the transformer
            decoder network

            max_nhead: the maximum number of heads a multi-head attention
            layer can have

            nhead: the number of heads multi-head attention layers have
            in the network

            dim_feedforward: the number of dimensions in the feature 
            vector in the feedforward layers of the network.

            nhead: the number of heads in the multi-head attention layers
            in the transformer network

        NOTE: This module uses a heuristic to determine the number of the
        heads in the multihead attention layers in the transformer network

        NOTE: Please see ModuleWrapper class for other built-in methods.
    """
    def __init__(self,
                 model_name : str,
                 input_size : int,
                 num_enc_layers : int,
                 num_dec_layers : int,
                 max_nhead : int = 6,
                 dim_feedforward : int = 2048,
                 dropout_prob : float = 0.1,
                 activation : str = 'relu',
                 device : torch.device = None):
        """ Inits a Transformer instance             
            Please see https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer
            as a reference to the model's transformer network

            Raises:
                Exception: if the input size is not divisible by a number
                between 2-13, because this is required to determine the 
                number of heads in the multihead attention layers in 
                the transformer network
        """
        super().__init__(model_name + "_transformer", device = device)

        self.input_size = input_size
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.dim_feedforward = dim_feedforward
        self.max_nhead = max_nhead

        self.nhead = 0
        for i in range(max_nhead,1,-1):
            if self.input_size % i == 0 and i > self.nhead:
                self.nhead = i

        assert self.nhead != 0, ("Please change the range of values that "
                                 + "are tested to determine nhead (number"
                                 + "of heads in multihead attention layers"
                                 + "in the transformer network)")

        self.model = nn.Transformer(self.input_size,
                                    nhead=self.nhead,
                                    num_encoder_layers=self.num_enc_layers,
                                    num_decoder_layers=self.num_dec_layers,
                                    dim_feedforward=self.dim_feedforward,
                                    dropout=dropout_prob,
                                    activation=activation)

    def forward(self,
                source : torch.Tensor,
                targ : torch.Tensor,
                src_key_padding_mask : torch.Tensor = None,
                tgt_key_padding_mask : torch.Tensor = None
                ) -> torch.Tensor:
        """ Forward pass through the model """
        return self.model(source,
                          targ,
                          src_key_padding_mask = src_key_padding_mask,
                          tgt_key_padding_mask = tgt_key_padding_mask)

class TransformerComparer(ModuleWrapper):
    """ A ModuleWrapper Class with a model composed of a transformer
        decoder network, which I have used in the past to compare
        long time series to themselves.

        Additional Attributes:
            input_size: an integer describing the number of dimensions
            in an input feature vector

            num_layers: an integer describing the desired number of layers
            in the transformer decoder network

            dim_feedforward: the number of dimensions in the feature 
            vector in the feedforward layers of the network.

            droput: a boolean indicating whether to include dropout
            layers in the deCNN

            dropout_prob: the dropout probability to use during training

            max_nhead: the maximum number of heads a multi-head attention
            layer can have

            nhead: the number of heads multi-head attention layers have
            in the network

            uc: a boolean indicating whether to keep drop out layers
            active during evalution

            layer_name_list: a list of strings with a name for each
            layer in the network module

        NOTE: This module uses a heuristic to determine the size of the
        intermediate layers in the network

        NOTE: Please see ModuleWrapper class for other built-in methods.
    """
    def __init__(self,
                 model_name : str,
                 input_size : int,
                 num_layers : int,
                 dropout_prob : float = 0.1,
                 dropout_bool : bool = True,
                 max_nhead : int = 9,
                 uc : bool = True,
                 dim_feedforward : int = 128,
                 device : torch.device = None):
        """ Inits a TransformerComparer instance             
            Please see https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html#torch.nn.TransformerDecoderLayer
            as a reference to the model's transformer decoder layers

            Raises:
                Exception: if the input size is not divisible by a number
                between 2-13, because this is required to determine the 
                number of heads in the multihead attention layers in 
                the transformer network
        """
        super().__init__(model_name + "_trans_comparer", device = device)
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.dropout = dropout_bool
        self.dim_feedforward = dim_feedforward
        self.uc = uc

        # print("Dropout Rate: ", self.dropout_prob)

        self.nhead = 0
        self.max_nhead = max_nhead
        for i in range(max_nhead,1,-1):
            if self.input_size % i == 0 and i > self.nhead:
                self.nhead = i

        assert self.nhead != 0, ("Please change the range of values that "
                                 + "are tested to determine nhead (number"
                                 + "of heads in multihead attention layers"
                                 + "in the transformer network)")

        layer_list = []
        self.layer_name_list = []

        for i in range(self.num_layers):
            if dropout_bool:
                layer_list.append(nn.Dropout(p=dropout_prob))
                self.layer_name_list.append('dropout1d_' + str(i))

            layer_list.append(nn.TransformerDecoderLayer(self.input_size,
                                                         self.nhead,
                                                         dim_feedforward = self.dim_feedforward))

            self.layer_name_list.append('trans_dec_' + str(i))

        self.model = nn.ModuleList(layer_list)

    def forward(self,
                seq : torch.Tensor,
                padding_mask : torch.Tensor = None) -> torch.Tensor:
        """ Forward pass through the model """
        # print("Padding mask size: ", mem_padding_mask.size())
        # print("input size: ", tgt_seq.size())
        for i, layer in enumerate(self.model):
            name = self.layer_name_list[i]
            if name[:4] == 'drop':
                if self.uc:
                    layer.train()
                    
                if i == 0:
                    out = layer(seq)
                else:
                    out = layer(out)
            else:
                if i == 0:
                    out = layer(seq,
                                seq,
                                memory_key_padding_mask = padding_mask,
                                tgt_key_padding_mask = padding_mask)
                else:
                    out = layer(out,
                                out,
                                memory_key_padding_mask = padding_mask,
                                tgt_key_padding_mask = padding_mask)
        return out

###################################################
### Models Composed of Multiple Neural Networks ###
###################################################
class ModelWrapper(nn.Module):
    """ a super class / wrapper for all models composed of multiple 
        neural network modules with integrated methods for loading, 
        saving and returning the parameters in the model.

        Attributes:
            model_name: a string with the model name of neural network
            module
            device: a torch.device that the module performs its
            calculations on
            model: the multi-layer neural network module which performs
            the calculations
    """
    def __init__(self,
                 model_name : str,
                 device : torch.device = None,
                #  parallel : bool = False
                ):
        """ Inits a ModelWrapper Instance """
        super().__init__()
        self.model_name = model_name   
        self.device = device
        # self.parallel = False
    
    def set_device(self, device):
        """ Sets the device of the modules within the model """
        self.device = device
        for model in self._modules.values():
            model.set_device(self.device)
    
    def save(self, epoch_num : int, model_dir : str):
        """ Saves all the neural network modules in the model to a 
            directory
        """
        for model in self._modules.values():
            model.save(epoch_num, model_dir)

    def load(self, epoch_num : int, model_dir : str):
        """ Loads all the neural network modules in the model from a 
            directory
        """
        for model in self._modules.values():
            model.load(epoch_num, model_dir)

    def set_uc(self, uc_bool : bool):
        """ Sets the instance attribute 'uc' in the neural network 
            modules if they have that attribute.
        """
        for model in self._modules.values():  
            if hasattr(model, 'uc'):
                model.uc = uc_bool
            elif hasattr(model, 'set_uc'):
                model.set_uc(uc_bool)

    def parameters(self) -> List[torch.Tensor]:
        """ Returns the weights / parameters of all the modules in the
            model
        """
        parameters = []
        for model in self._modules.values():
            parameters += list(model.parameters())

        return parameters

    def train(self):
        """ Sets all the neural network modules in the model to training
            mode
        """
        for model in self._modules.values():       
            model.train()

    def eval(self):
        """ Sets all the neural network modules in the model to
            evaluation mode
        """
        for model in self._modules.values():            
            model.eval()

    # TODO: add functionality of running models on multiple GPUs
    # def set_parallel(self, parallel_bool):
    #     if parallel_bool:
    #         for model in self.module_list:
    #             model.set_parallel(parallel_bool)

    #     self.parallel = parallel_bool

class ResNetFCN(ModelWrapper):
    """ A ModelWrapper class with a model composed of a set of
        fully-connected layers with skip connections between every two
        layers also known as a residual network

        Additional Attributes:
            input_size: an integer describing the number of dimensions
            in an input feature vector

            output_size: an integer describing the desired number of 
            dimensions in the models output vector

            num_layers: the number of layers in the fully-connected
            network

            droput: a boolean indicating whether to include dropout
            layers in the deCNN

            dropout_prob: the dropout probability to use during training

            uc: a boolean indicating whether to keep drop out layers
            active during evalution

            layer_name_list: a list of strings with a name for each
            layer in the network module

        NOTE: Please see ModuleWrapper class for other built-in methods.
    """
    def __init__(self,
                 model_name : str,
                 input_size : int,
                 output_size : int,
                 num_layers : int,
                 dropout : bool = True,
                 dropout_prob : float = 0.5,
                 uc : bool = False,
                 device : torch.device = None):
        """ Inits a ResNetFCN network

            Raises:
                Exception if the input_size is smaller than the output_size
                or the input_size is smaller than 64, this is a heuristic
                that tries to make sure the model is large (has enough
                parameters) to learn something meaningful.
        """
        super().__init__(model_name, device)
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        assert (input_size > output_size) or (input_size >= 64)

        for i in range(self.num_layers):
            key = "{}_reslayer_{}".format(model_name,str(i + 1).zfill(4))

            if i == self.num_layers - 1:
                self._modules[key] = FCN(key,
                                         self.input_size,
                                         self.output_size,
                                         1,
                                         nonlinear = False,
                                         batchnorm = False,
                                         dropout = dropout,
                                         dropout_prob = dropout_prob,
                                         uc = uc,
                                         device = self.device).to(self.device)
            else:
                self._modules[key] = FCN(key,
                                         self.input_size,
                                         self.input_size,
                                         1,
                                         nonlinear = True,
                                         batchnorm = False,
                                         dropout = dropout,
                                         dropout_prob = dropout_prob,
                                         uc = uc,
                                         device = self.device).to(self.device)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """ Forward pass through the model """
        for key, model in self._modules.items():
            i = int(key[-4:]) - 1
            if i == 0 and self.num_layers == 1:
                output = model(x)
            elif i == 0 and self.num_layers != 1:
                output = model(x) + x
                residual = output.clone()

            elif i == len(self._modules.keys()) - 1:
                output = model(output)
            else:
                output = model(output) + residual
                residual = output.clone()
                
        return output



