import torch
import torch.nn as nn
from torch.nn import functional as F 
from torch.distributions import Normal
from torch.distributions import Categorical
import torchvision
import copy
import yaml
import numpy as np

from utils import *

'''
Features to add

DeCNN network
Network for point clouds
Network for video
Network for sequential point clouds
Sequential data deconv
add dilation to your CNNs
feature for Training on multiple GPUs
'''

#########################################
# Params class for learning specific parameters
#########################################
class Params_module(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.p = nn.Parameter(nn.init.uniform_(torch.empty(size)))
    def forward(self):
        return self.p

class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.view(input.size(0), -1)


class TransformerDecoderLayer_Modified(nn.Module):
    '''TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    '''

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, uc = True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.uc = uc

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        if self.uc:
            self.dropout1.train()
            self.dropout2.train()
            self.dropout3.train()
            self.self_attn.train()
            self.multihead_attn.train()

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)

        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
#########################################
# Current Model Types Supported 
########################################
'''
All models have four other methods

1. init - initializes the network with the inputs requested by the user
2. forward - returns the output of the network for a given input
3. save - saves the model
4. load - loads a model if there is a nonempty path corresponding to that model in the yaml file
5. weight_parameters - returns weight parameters of module
6. bias_parameters - returns bias parameters of module

'''
#### super class of all models for logging and loading models
class Proto_Model(nn.Module):
    def __init__(self, save_name, load_name, device):
        super().__init__()
        self.save_name = save_name
        self.load_name = load_name
        self.device = device
        # self.parallel = False

    def forward(self, inputs):
        return self.model(inputs)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def save(self, epoch_num):
        ckpt_path = '{}_{}'.format(self.save_name, epoch_num)
        print("Saved Model to: ", ckpt_path)
        torch.save(self.model.state_dict(), ckpt_path)

    def load(self, epoch_num):
        ckpt_path = '{}_{}'.format(self.load_name, epoch_num)
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt)
        print("Loaded Model to: ", ckpt_path)

    # def set_parallel(self, parallel_bool):
    #     if parallel_bool:
    #        self.model =  nn.DataParallel(self.model)

    #     self.parallel = parallel_bool

class Params(Proto_Model):
    def __init__(self, save_name, load_name, size, device= None):
        super().__init__(save_name + "_params", load_name + "_params", device = device)

        self.device = device
        self.size = size
        self.model = Params_module(self.size)

    def forward(self):
        return self.model()
    # def set_parallel(self, bool):
    #     pass

class CONV2DN(Proto_Model):
    def __init__(self, save_name, load_name, input_size, output_size, nonlinear = False,\
        batchnorm = True, dropout = False, dropout_prob = 0.5, uc = True, device = None):
        super().__init__(save_name + "_cnn2d", load_name + "_cnn2d", device = device)

        self.input_size = input_size
        self.output_size = output_size

        self.batchnorm = batchnorm
        self.nonlinear = nonlinear
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.uc = uc

        assert self.dropout != self.batchnorm

        #assume that the prime factorization of rows and cols is composed of only powers of 3 and 2
        e_p_list = get_2Dconv_params(input_size, output_size) # encoder parameters

        # print(e_p_list)
        layer_list = []
        self.layer_name_list = []

        for i, e_p in enumerate(e_p_list):
            layer_list.append(nn.Conv2d(e_p[0] , e_p[1], kernel_size=(e_p[2], e_p[3]),\
                stride=(e_p[4], e_p[5]), padding=(e_p[6], e_p[7]), bias=True))
            self.layer_name_list.append('conv2d_' + str(i))

            if i != (len(e_p_list) - 1) or nonlinear:
                if dropout:
                    layer_list.append(nn.Dropout(p=dropout_prob))
                    self.layer_name_list.append('dropout_' + str(i))

                if batchnorm:
                    layer_list.append(nn.BatchNorm2d(e_p[1]))
                    self.layer_name_list.append('batchnorm2d_' + str(i))

                layer_list.append(nn.LeakyReLU(0.1, inplace = False))
                self.layer_name_list.append('leaky_relu_' + str(i))



        self.model = nn.ModuleList(layer_list)

        # -----------------------
        # weight initialization
        # -----------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
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
class DECONV2DN(Proto_Model):
    def __init__(self, save_name, load_name, input_size, output_size, nonlinear=False,\
     batchnorm = True, dropout = False, dropout_prob = 0.5, uc = False, device = None):
        super().__init__(save_name + "_dcnn2d", load_name + "_dcnn2d", device = device)

        self.input_size = input_size
        self.output_size = output_size

        self.batchnorm = batchnorm
        self.nonlinear = nonlinear
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.uc = uc

        assert self.dropout != self.batchnorm

        #assume that the prime factorization of rows and cols is composed of only powers of 3 and 2
        e_p_list = get_2Dconv_params(output_size, input_size) # encoder parameters
        e_p_list.reverse()

        layer_list = []
        self.layer_name_list = []

        for i, e_p in enumerate(e_p_list):
            layer_list.append(nn.ConvTranspose2d(e_p[1] , e_p[0], kernel_size=(e_p[2], e_p[3]),\
                stride=(e_p[4], e_p[5]), padding=(e_p[6], e_p[7]), bias=True))
            self.layer_name_list.append('deconv2d_' + str(i))

            if i != (len(e_p_list) - 1) or nonlinear:
                if dropout:
                    layer_list.append(nn.Dropout(p=dropout_prob))
                    self.layer_name_list.append('dropout_' + str(i))

                if batchnorm:
                    layer_list.append(nn.BatchNorm2d(e_p[1]))
                    self.layer_name_list.append('batchnorm2d_' + str(i))

                layer_list.append(nn.LeakyReLU(0.1, inplace = False))
                self.layer_name_list.append('leaky_relu_' + str(i))


        self.model = nn.ModuleList(layer_list)
        # -----------------------
        # weight initialization
        # -----------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        for i, layer in enumerate(self.model):
            name = self.layer_name_list[i]
            if name[:4] == 'drop' and self.uc:
                layer.train()

            if i == 0:
                out = layer(inputs)
            else:
                out = layer(out)

        return out

#### a time series network
class CONV1DN(Proto_Model):
    def __init__(self, save_name, load_name, input_size, output_size, nonlinear = False,\
        batchnorm = True, dropout = False, dropout_prob = 0.5, uc = False, device = None):
        super().__init__(save_name + "_cnn1d", load_name + "_cnn1d", device = device)

        self.input_size = input_size
        self.output_size = output_size

        self.batchnorm = batchnorm
        self.nonlinear = nonlinear
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.uc = uc

        assert self.dropout != self.batchnorm

        # print("Dropout Rate: ", self.dropout_prob)
        #assume that the prime factorization of rows and cols is composed of only powers of 3 and 2
        e_p_list = get_1Dconv_params(input_size, output_size) # encoder parameters

        layer_list = []
        self.layer_name_list = []

        for i, e_p in enumerate(e_p_list):
            layer_list.append(nn.Conv1d(e_p[0] , e_p[1], kernel_size= e_p[2],\
                stride=e_p[3], padding=e_p[4], bias=True))
            self.layer_name_list.append('conv1d_' + str(i))

            if i != (len(e_p_list) - 1) or nonlinear:
                if dropout:
                    layer_list.append(nn.Dropout(p=dropout_prob))
                    self.layer_name_list.append('dropout_' + str(i))

                if batchnorm:
                    layer_list.append(nn.BatchNorm1d(e_p[1]))
                    self.layer_name_list.append('batchnorm1d_' + str(i))

                layer_list.append(nn.LeakyReLU(0.1, inplace = False))
                self.layer_name_list.append('leaky_relu_' + str(i))
               
        self.model = nn.ModuleList(layer_list)

    def forward(self, inputs):
        for i, layer in enumerate(self.model):
            name = self.layer_name_list[i]
            if name[:4] == 'drop' and self.uc:
                layer.train()

            if i == 0:
                out = layer(inputs)
            else:
                out = layer(out)

        return out

#### a 1D deconvolutional network
class DECONV1DN(Proto_Model):
    def __init__(self, save_name, load_name, input_size, output_size, nonlinear = False,\
        batchnorm = True, dropout = False, dropout_prob = 0.5, uc = False, device = None):
        super().__init__(save_name + "_cnn1d", load_name + "_cnn1d", device = device)

        self.input_size = input_size
        self.output_size = output_size

        self.batchnorm = batchnorm
        self.nonlinear = nonlinear
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.uc = uc

        assert self.dropout != self.batchnorm

        e_p_list = get_1Dconv_params(input_size, output_size) # encoder parameters
        e_p_list.reverse()

        layer_list = []
        self.layer_name_list = []

        for i, e_p in enumerate(e_p_list):
            layer_list.append(nn.ConvTranspose1d(e_p[0] , e_p[1], kernel_size= e_p[2],\
                stride=e_p[3], padding=e_p[4], bias=True))
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

    def forward(self, inputs):
        for i, layer in enumerate(self.model):
            name = self.layer_name_list[i]
            if name[:4] == 'drop' and self.uc:
                layer.train()

            if i == 0:
                out = layer(inputs)
            else:
                out = layer(out)

        return out

#### a fully connected network
class FCN(Proto_Model):
    def __init__(self, save_name, load_name, input_channels, output_channels, num_layers,
        nonlinear = False, batchnorm = True, dropout = False, dropout_prob = 0.5, uc = False, device = None):
        super().__init__(save_name + "_fcn", load_name + "_fcn", device = device)

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.num_layers = num_layers
        self.batchnorm = batchnorm
        self.nonlinear = nonlinear
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.uc = uc

        assert self.batchnorm != self.dropout

        # print("Dropout Rate: ", self.dropout_prob)
        # -----------------------
        # Fully connected network
        # -----------------------
        layer_list = []
        self.layer_name_list = []

        for i in range(self.num_layers):
            if dropout:
                layer_list.append(nn.Dropout(p=dropout_prob))
                self.layer_name_list.append('dropout1d_' + str(i))

            if i == 0:
                layer_list.append(nn.Linear(input_channels, output_channels))
            else:
                layer_list.append(nn.Linear(output_channels, output_channels))

            self.layer_name_list.append('linear_' + str(i))

            if i != (self.num_layers - 1) or nonlinear:
                if batchnorm:
                    layer_list.append(nn.BatchNorm1d(e_p[1]))
                    self.layer_name_list.append('batchnorm1d_' + str(i))

                layer_list.append(nn.LeakyReLU(0.1, inplace = False))
                self.layer_name_list.append('leaky_relu_' + str(i))
               
        self.model = nn.ModuleList(layer_list)

    def forward(self, inputs):
        for i, layer in enumerate(self.model):
            name = self.layer_name_list[i]
            if name[:4] == 'drop' and self.uc:
                layer.train()

            if i == 0:
                out = layer(inputs)
            else:
                out = layer(out)

        return out

### a basic recurrent neural network 
class RNNCell(Proto_Model):
    def __init__(self, save_name, load_name, input_channels, output_channels, nonlinearity = 'tanh',  device = None):
        super().__init__(save_name + "_rnn", load_name + "_rnn", device = device)

        self.device = device
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.num_layers = num_layers
        # -----------------------
        # Recurrent neural network
        # -----------------------
        self.model = nn.RNNCell(self.input_channels, self.output_channels, nonlinearity = nonlinearity)

    def forward(self, x, h = None):
        if h is None:
            h = torch.zeros((x.size(0), self.output_channels))

        return self.model(x, h)

### a gated recurrent neural network
class GRUCell(Proto_Model):
    def __init__(self, save_name, load_name, input_channels, output_channels, device = None):
        super().__init__(save_name + "_gru", load_name + "_gru", device = device)

        self.device = device
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.num_layers = num_layers
        # -----------------------
        # Recurrent neural network
        # -----------------------
        self.model = nn.GRUCell(self.input_channels, self.output_channels)

    def forward(self, x, h = None):
        if h is None:
            h = torch.zeros((x.size(0), self.output_channels))
        return self.model(x, h)

### a long short term memory recurrent neural network
class LSTMCell(Proto_Model):
    def __init__(self, save_name, load_name, input_channels, output_channels, device = None):
        super().__init__(save_name + "_lstm", load_name + "_lstm", device = device)

        self.device = device
        self.input_channels = input_channels
        self.output_channels = output_channels
        # -----------------------
        # Recurrent neural network
        # -----------------------

        self.model = nn.LSTMCell(self.input_channels, self.output_channels)

    def forward(self, x, h = None, c=None):
        if h is None or c is None:
            h = torch.zeros(x.size(0), self.output_channels).to(self.device)
            c = torch.zeros(x.size(0), self.output_channels).to(self.device)

        return self.model(x, (h, c))

# class Transformer(Proto_Model):
class Transformer(Proto_Model):
    def __init__(self, save_name, load_name, input_size, num_enc_layers, num_dec_layers,\
     norm = None, nhead = 8, dim_feedforward = 2048, dropout = 0.1, activation = 'relu', device = None):
        super().__init__(save_name + "_transformer", load_name + "_transformer", device = device)

        self.device = device
        self.input_size = input_size
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers

        self.model = nn.Transformer(self.input_size, nhead=nhead, num_encoder_layers=self.num_enc_layers,\
         num_decoder_layers=self.num_dec_layers, dim_feedforward=dim_feedforward,\
          dropout=dropout, activation=activation)

    def forward(self, source, targ):
        return self.model(source, targ)

class Transformer_Encoder(Proto_Model):
    def __init__(self, save_name, load_name, input_size, num_layers,\
     norm = None, nhead = 8, dim_feedforward = 2048, dropout = 0.1, activation = 'relu', device = None):
        super().__init__(save_name + "_trans_encoder", load_name + "_trans_encoder", device = device)
        self.device = device
        self.input_size = input_size
        self.num_layers = num_layers
        self.norm = norm
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation

        self.model = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.input_size, self.nhead, dim_feedforward = self.dim_feedforward, dropout = self.dropout,\
            activation = self.activation), num_layers = self.num_layers, norm = self.norm)

class Transformer_Comparer(Proto_Model):
    def __init__(self, save_name, load_name, input_size, num_layers, dropout_prob = 0.1, uc = True,\
     nhead = 8, dim_feedforward = 128, activation = 'relu', device = None):
        super().__init__(save_name + "_trans_decoder", load_name + "_trans_decoder", device = device)
        self.device = device
        self.input_size = input_size
        self.num_layers = num_layers

        self.dropout_prob = dropout_prob
        self.uc = uc

        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.uc = uc

        # print("Dropout Rate: ", self.dropout_prob)

        layer_list = []
        self.layer_name_list = []

        for i in range(self.num_layers):
            layer_list.append(TransformerDecoderLayer_Modified(self.input_size, self.nhead, dim_feedforward = self.dim_feedforward,\
                dropout = self.dropout_prob, uc =  self.uc))
            self.layer_name_list.append('trans_dec_' + str(i))

        self.model = nn.ModuleList(layer_list)

    def forward(self, seq, padding_mask = None):
        # print("Padding mask size: ", mem_padding_mask.size())
        # print("input size: ", tgt_seq.size())
        for i, layer in enumerate(self.model):
            name = self.layer_name_list[i]
            if i == 0:
                out = layer(seq, seq, memory_key_padding_mask = padding_mask, tgt_key_padding_mask = padding_mask)
            else:
                out = layer(out, out, memory_key_padding_mask = padding_mask, tgt_key_padding_mask = padding_mask)

        return out
######################################
# Current Macromodel Types Supported
#####################################
'''
All macromodels have four main methods

1. init - initializes the network with the inputs requested by the user
2. forward - returns the output of the network for a given input
3. save - saves the model
4. load - loads a model if there is a nonempty path corresponding to that model in the yaml file

Note the ProtoMacromodel below does not have a forward method, this must defined by the subclasses which 
call the ProtoMacromodel as a superclass
'''
#### super class for models of models for logging and loading models
class Proto_Macromodel(nn.Module):
    def __init__(self):
        super().__init__()   
        self.model_list = []
        self.parallel = False

    def save(self, epoch_num):
        for model in self.model_list:
            model.save(epoch_num)

    def load(self, epoch_num):
        for model in self.model_list:
            model.load(epoch_num)

    def parameters(self):
        parameters = []
        for model in self.model_list:
            parameters += list(model.parameters())
        return parameters

    def eval(self):
        for model in self.model_list:
            model.eval()

    # def set_parallel(self, parallel_bool):
    #     if parallel_bool:
    #         for model in self.model_list:
    #             model.set_parallel(parallel_bool)

    #     self.parallel = parallel_bool

class ResNetFCN(Proto_Macromodel):
    def __init__(self, save_name, load_name, input_channels, output_channels, num_layers, dropout = True, dropout_prob = 0.5, uc = False, device = None):
        super().__init__()
        self.device = device
        self.input_channels = input_channels
        self.save_bool = True
        self.output_channels = output_channels

        self.num_layers = num_layers
        self.model_list = []

        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                self.model_list.append(FCN(save_name + "_layer_" + str(i + 1), load_name + "_layer_" + str(i + 1),\
                 self.input_channels, self.output_channels, 2, nonlinear = False, batchnorm = False, dropout = dropout,\
                 dropout_prob = dropout_prob, uc = uc, device = self.device).to(self.device))
            else:
                self.model_list.append(FCN(save_name + "_layer_" + str(i + 1), load_name + "_layer_" + str(i + 1),\
                 self.input_channels, self.input_channels, 2, nonlinear = False, batchnorm = False, dropout = dropout,\
                 dropout_prob = dropout_prob, uc = uc, device = self.device).to(self.device))

    def forward(self, x):
        for i, model in enumerate(self.model_list):
            if i == 0 and self.num_layers == 1:
                output = model(x)
            elif i == 0 and self.num_layers != 1:
                output = model(x) + x
                residual = output.clone()

            elif i == len(self.model_list) - 1:
                output = model(output)
            else:
                output = model(output) + residual
                residual = output.clone()
                
        return output

