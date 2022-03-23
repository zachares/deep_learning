import torch
from torch_geometric.nn import GAT

from deep_learning.models_manager.model_wrappers import ModuleWrapper

class GraphAttentionModule(ModuleWrapper):
    """ A module wrapper for a GAT network """
    def __init__(
        self,
        model_name: str,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        dropout_prob: float=0.1,
        dropout_bool: bool=True,
        device : torch.device=None
    ):
        super().__init__(model_name + "_graph_attention", device=device)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.dropout = dropout_bool

        self.layer_name_list = []

        self.layer_name_list.append('gat_model_')
        self.model = GAT(
            in_channels=self.input_size,
            hidden_channels=self.hidden_size,
            num_layers=self.num_layers,
            out_channels=self.output_size,
            dropout= 0.0 if not self.dropout else self.dropout_prob,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the model """
        return self.model(x=x, edge_index=edge_index)
