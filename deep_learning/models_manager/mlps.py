import torch
import torch.nn as nn

from deep_learning.models_manager.model_wrappers import ModuleWrapper, ModelWrapper

class FCN(ModuleWrapper):
    """ A ModuleWrapper Class with a model composed of a multi-layer
        fully connected layers
    """
    def __init__(
        self,
        model_name : str,
        input_size: int,
        hidden_size: int,
        output_size : int,
        num_layers : int,
        nonlinear : bool=False,
        batchnorm : bool=True,
        dropout : bool=False,
        dropout_prob : float=0.5,
        leak_rate: float=0.1,
        uc : bool=False,
        device : torch.device=None
    ):
        """ Inits a FCN instance
            Please see https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
            as a reference to the linear layers in the network
        """
        super().__init__(model_name + "_fcn", device = device)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batchnorm = batchnorm
        self.nonlinear = nonlinear
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.leak_rate = leak_rate
        self.uc = uc
        layer_list = []
        self.layer_name_list = []
        for i in range(self.num_layers):
            in_size = self.input_size if i == 0 else self.hidden_size
            out_size = self.hidden_size if i != (self.num_layers - 1) else self.output_size
            if dropout and i != 0:
                layer_list.append(nn.Dropout(p=dropout_prob))
                self.layer_name_list.append('dropout1d_' + str(i))
            self.layer_name_list.append('linear_' + str(i))
            layer_list.append(nn.Linear(in_size, out_size))
            if i != (self.num_layers - 1) or nonlinear:
                if batchnorm:
                    layer_list.append(nn.BatchNorm1d(out_size))
                    self.layer_name_list.append('batchnorm1d_' + str(i))
                layer_list.append(nn.LeakyReLU(self.leak_rate, inplace=False))
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


class ResNetFCN(ModelWrapper):
    """ A ModelWrapper class with a model composed of a set of
        fully-connected layers with skip connections between every two
        layers also known as a residual network
    """
    def __init__(
        self,
        model_name : str,
        input_size : int,
        hidden_size : int,
        output_size : int,
        num_layers : int,
        dropout : bool=True,
        dropout_prob : float=0.5,
        uc : bool=False,
        device : torch.device=None
    ):
        """ Inits a ResNetFCN network
        """
        super().__init__(model_name, device)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        for i in range(self.num_layers):
            key = "{}_reslayer_{}".format(model_name,str(i + 1).zfill(4))
            in_size = self.input_size if i == 0 else self.hidden_size
            out_size = self.hidden_size if i != (self.num_layers - 1) else self.output_size
            nonlinear = False if i == self.num_layers - 1 else True
            self._modules[key] = FCN(
                key,
                input_size=in_size,
                hidden_size=out_size,
                output_size=out_size,
                num_layers=1,
                nonlinear=nonlinear,
                batchnorm=False,
                dropout=dropout,
                dropout_prob=dropout_prob,
                uc=uc,
                device=self.device
            ).to(self.device)
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
