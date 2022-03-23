import numpy as np
import torch
import torch.nn as nn

from typing import List, Tuple

from deep_learning.models_manager.model_wrappers import ModuleWrapper

class CONV2DN(ModuleWrapper):
    """ A ModuleWrapper Class with a model composed of a multi-layer
        2d convolutional neural network.
    """
    def __init__(
        self,
        model_name : str,
        input_size : Tuple[int],
        output_size : Tuple[int],
        nonlinear : bool = False,
        batchnorm : bool = True,
        dropout : bool = False,
        dropout_prob : float = 0.5,
        uc : bool = True,
        device : torch.device = None
    ):
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
        e_p_list = get_2Dconv_params(input_size, output_size) # encoder parameters

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
            out = layer(inputs) if i == 0 else layer(out)
        return out

#### a 2D deconvolutional network
class DECONV2DN(ModuleWrapper):
    """ A ModuleWrapper Class with a model composed of a multi-layer
        2d deconvolutional neural network.
    """
    def __init__(
        self,
        model_name : str,
        input_size : Tuple[int],
        output_size : Tuple[int],
        nonlinear : bool=False,
        batchnorm : bool=True,
        dropout : bool=False,
        dropout_prob : float=0.5,
        uc : bool=False,
        device : torch.device=None
    ):
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
        e_p_list = get_2Dconv_params(output_size, input_size)
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
            out = layer(inputs) if i == 0 else layer(out)
        return out


class CONV1DN(ModuleWrapper):
    """ A ModuleWrapper Class with a model composed of a multi-layer
        1d convolutional neural network. This network can be useful
        for processing short time series (num_steps < 100) of feature
        vectors
    """
    def __init__(
        self,
        model_name : str,
        input_size : Tuple[int],
        output_size : Tuple[int],
        nonlinear : bool=False,
        batchnorm : bool=True,
        dropout : bool=False,
        dropout_prob : float=0.5,
        uc : bool=False,
        device : torch.device=None
    ):
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

        # Calculates the number and size of each layer based on the input
        # and output size
        e_p_list = get_1Dconv_params(input_size, output_size)

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
            out = layer(inputs) if i == 0 else layer(out)
        return out


class DECONV1DN(ModuleWrapper):
    """ A ModuleWrapper Class with a model composed of a multi-layer
        1d deconvolutional neural network. This network can be useful
        for generating short time series (num_steps < 100) of feature
        vectors
        NOTE: Please see ModuleWrapper class for other built-in methods.
    """
    def __init__(
        self,
        model_name : str,
        input_size : Tuple[int],
        output_size : Tuple[int],
        nonlinear : bool=False,
        batchnorm : bool=True,
        dropout : bool=False,
        dropout_prob : float=0.5,
        uc : bool=False,
        device : torch.device=None
    ):
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
        # Calculates the number and size of each layer based on the input
        # and output size
        e_p_list = get_1Dconv_params(input_size, output_size)
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
            out = layer(inputs) if i == 0 else layer(out)
        return out


def get_2Dconv_params(input_size : Tuple[int],
                      output_size : Tuple[int]) -> List[Tuple[int]]:
    """ Calculates the parameters for each layer of a 2D convolutional
        neural network (CNN) given a tuple describing the input size of
        data points 'input_size' and a tuple describing the desired
        output size of data points 'output_size'

        NOTE: this function assumes input height and width have only
        prime factors 2, 3, 5, 7 for now and the input height and width
        are divisible by their respective output height and width

        Returns:
            A list of tuples where each tuple in the list contains
            the parameters for a layer in the CNN

        Raises:
            ValueError: if input height or width are not divisble by
            their respective output height and width

    """
    input_chan, input_height, input_width = input_size
    output_chan, output_height, output_width = output_size

    chan_list = get_chan_list(input_chan, output_chan)

    if input_height % output_height == 0 and input_width % output_width == 0:
        prime_fact_rows = get_prime_fact(input_height // output_height)
        prime_fact_cols = get_prime_fact(input_width // output_width)
    else:
        raise ValueError("The input height and width are not divisible"
                        + " by the output height and width")
    # print(prime_fact_rows)
    # print(prime_fact_cols)

    if len(prime_fact_cols) > len(prime_fact_rows):

        while len(prime_fact_cols) > len(prime_fact_rows):
            prime_fact_rows.append(1)

    elif len(prime_fact_cols) < len(prime_fact_rows):

        while len(prime_fact_cols) < len(prime_fact_rows):
            prime_fact_cols.append(1)

    if len(prime_fact_cols) > len(chan_list):

        while len(prime_fact_cols) > len(chan_list):
            chan_list.append(chan_list[-1])

    elif len(prime_fact_cols) < len(chan_list):

        idx = 1

        while len(prime_fact_cols) < len(chan_list):

            prime_fact_cols.insert(idx, 1)
            prime_fact_rows.insert(idx, 1)

            idx += 2

            if idx >= len(prime_fact_cols):

                idx = 1

    e_p = np.zeros((8,len(prime_fact_rows))).astype(np.int16)

    chan_list.append(chan_list[-1])

    for idx in range(len(prime_fact_rows)):

        # first row input channel
        e_p[0, idx] = chan_list[idx]
        # second row output channel
        e_p[1, idx] = chan_list[idx + 1]
        # third row row kernel
        # fifth row row stride
        if prime_fact_rows[idx] == 7:
            e_p[2,idx] = 9
            e_p[4,idx] = 7
        elif prime_fact_rows[idx] == 5:
            e_p[2,idx] = 7
            e_p[4,idx] = 5
        elif prime_fact_rows[idx] == 3:
            e_p[2,idx] = 5
            e_p[4,idx] = 3
        elif prime_fact_rows[idx] == 2:
            e_p[2,idx] = 4
            e_p[4,idx] = 2
        else:
            e_p[2,idx] = 3
            e_p[4,idx] = 1
        # fourth row col kernel
        # sixth row col stride
        if prime_fact_cols[idx] == 7:
            e_p[3,idx] = 9
            e_p[5,idx] = 7
        elif prime_fact_cols[idx] == 5:
            e_p[3,idx] = 7
            e_p[5,idx] = 5
        elif prime_fact_cols[idx] == 3:
            e_p[3,idx] = 5
            e_p[5,idx] = 3
        elif prime_fact_cols[idx] == 2:
            e_p[3,idx] = 4
            e_p[5,idx] = 2
        else:
            e_p[3,idx] = 3
            e_p[5,idx] = 1

        # seventh row row padding
        e_p[6, idx] = 1
        # eighth row col padding
        e_p[7,idx] = 1

    e_p_list = []

    for idx in range(e_p.shape[1]):

        e_p_list.append(tuple(e_p[:,idx]))

    return e_p_list


def get_1Dconv_params(input_size : Tuple[int], output_size : Tuple[int]) -> List[Tuple[int]]:
    """ Calculates the parameters for each layer of a 1D convolutional
        neural network (1DCNN) given a tuple describing the input size of
        data points 'input_size' and a tuple describing the desired
        output size of data points 'output_size'

        NOTE: this function assumes input height and width have only
        prime factors 2, 3, 5, 7 for now and the input width
        are divisible by its respective output width

        Returns:
            A list of tuples where each tuple in the list contains
            the parameters for a layer in the 1DCNN

        Raises:
            ValueError: if input height or width are not divisble by
            their respective output height and width

    """
    input_chan, input_width = input_size
    output_chan, output_width = output_size

    chan_list = get_chan_list(input_chan, output_chan)

    if input_width % output_width == 0:
        prime_fact= get_prime_fact(input_width // output_width)
    else:
        raise ValueError("The input  width is not divisible"
                        + " by the output width")

    if len(prime_fact) > len(chan_list):

        while len(prime_fact) > len(chan_list):
            chan_list.append(chan_list[-1])

    elif len(prime_fact) < len(chan_list):

        idx = 1

        while len(prime_fact) < len(chan_list):

            prime_fact.insert(idx, 1)

            idx += 2

            if idx >= len(prime_fact):

                idx = 1

    e_p = np.zeros((5,len(prime_fact))).astype(np.int16) #encoder parameters

    chan_list.append(chan_list[-1])

    for idx in range(len(prime_fact)):

        # print(prime_fact[idx])

        # first row input channel
        e_p[0, idx] = chan_list[idx]
        # second row output channel
        e_p[1, idx] = chan_list[idx + 1]
        # third row row kernel
        # fifth row row stride
        if prime_fact[idx] == 7:
            e_p[2,idx] = 9
            e_p[3, idx] = 7
        elif prime_fact[idx] == 5:
            e_p[2, idx] = 7
            e_p[3,idx] = 5
        elif prime_fact[idx] == 3:
            e_p[2,idx] = 5
            e_p[3, idx] = 3
        elif prime_fact[idx] == 2:
            e_p[2,idx] = 4
            e_p[3, idx] = 2
        else:
            e_p[2,idx] = 3
            e_p[3, idx] = 1
        # seventh row row padding
        e_p[4, idx] = 1

    e_p_list = []

    for idx in range(e_p.shape[1]):

        e_p_list.append(tuple(e_p[:,idx]))

    return e_p_list


def get_prime_fact(num : int) -> List[int]:
    """ Calculates a list of prime factors for an integer, but only
        supports integers which only have the prime factors 7, 5, 3, 2
        If a prime factor appears twice in the numbers factorization,
        it appears twice in the list that is returned.

        Raises:
            ValueError: if the integer has a prime factor other than
            7,5,3 or 2.
    """
    temp_num = num
    prime_fact_list = []

    while temp_num != 1:
        if temp_num % 7 == 0:
            temp_num = temp_num / 7
            prime_fact_list.append(7)
        elif temp_num % 5 == 0:
            temp_num = temp_num / 5
            prime_fact_list.append(5)
        elif temp_num % 3 == 0:
            temp_num = temp_num / 3
            prime_fact_list.append(3)
        elif temp_num % 2 == 0:
            temp_num = temp_num / 2
            prime_fact_list.append(2)
        else:
            raise ValueError("unsupported number " + str(num))

    if len(prime_fact_list) == 0:
        return []
    else:
        prime_fact_list.sort()
        prime_fact_list.reverse()

        return prime_fact_list


def get_chan_list(input_size : int, output_size : int) -> List[int]:
    """ Calculates a list of channel sizes for a convolutional neural
        network's layers

        Args:
            input_size: an integer describing the network's first layer's
            channel size
            output_size: an integer describing the network's last layer's
            desired channel size
    """
    chan_list = [input_size]

    if input_size > output_size:
        chan_list = get_chan_list(output_size, input_size)
        chan_list.reverse()
        return chan_list

    elif input_size < output_size:
        if output_size % input_size == 0:
            prime_factors = get_prime_fact(output_size // input_size)
        else:
            chan_list.append(input_size)
            prime_factors = get_prime_fact(output_size)
            chan_list.append(prime_factors[0])
            prime_factors = prime_factors[1:]

        for factor in prime_factors:
            chan_list.append(chan_list[-1] * factor)

        return chan_list
    else:
        return [input_size, output_size]
