import torch
import torch.nn as nn

from deep_learning.models_manager.model_wrappers import ModuleWrapper


class Transformer(ModuleWrapper):
    """ A ModuleWrapper Class with a model composed of transformer network
    """
    def __init__(
        self,
        model_name : str,
        input_size : int,
        num_enc_layers : int,
        num_dec_layers : int,
        max_nhead : int=6,
        dim_feedforward : int=2048,
        dropout_prob : float=0.1,
        activation : str='relu',
        device : torch.device=None
    ):
        """ Inits a Transformer instance
            Please see https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer
            as a reference to the model's transformer network
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

    def forward(
        self,
        source : torch.Tensor,
        targ : torch.Tensor,
        src_key_padding_mask : torch.Tensor = None,
        tgt_key_padding_mask : torch.Tensor = None
    ) -> torch.Tensor:
        """ Forward pass through the model """
        return self.model(
            source,
            targ,
            src_key_padding_mask = src_key_padding_mask,
            tgt_key_padding_mask = tgt_key_padding_mask
        )


class TransformerEncoder(ModuleWrapper):
    """ A ModuleWrapper Class with a model composed of a transformer
        decoder network, which I have used in the past to compare
        long time series to themselves.
    """
    def __init__(
        self,
        model_name : str,
        input_size : int,
        hidden_size : int,
        num_layers : int,
        dropout_prob : float=0.1,
        dropout_bool : bool=True,
        max_nhead : int=9,
        uc : bool=True,

        device : torch.device=None
    ):
        """ Inits a TransformerComparer instance
            Please see https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html#torch.nn.TransformerDecoderLayer
            as a reference to the model's transformer decoder layers
        """
        super().__init__(model_name + "_trans_comparer", device = device)
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.dropout = dropout_bool
        self.hidden_size = hidden_size
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
            layer_list.append(nn.TransformerEncoderLayer(
                self.input_size,
                self.nhead,
                dim_feedforward=self.hidden_size)
            )
            self.layer_name_list.append('trans_dec_' + str(i))
        self.model = nn.ModuleList(layer_list)

    def forward(
        self,
        seq : torch.Tensor,
        padding_mask : torch.Tensor=None
    ) -> torch.Tensor:
        """ Forward pass through the model """
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
                    out = layer(seq, src_key_padding_mask=padding_mask)
                else:
                    out = layer(out, src_key_padding_mask=padding_mask)
        return out
