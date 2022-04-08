import enum
import torch
from torch_geometric.nn import GCN, GraphSAGE, GAT, GIN, PNA

from deep_learning.models_manager.model_wrappers import ModuleWrapper

class GNNLoader(enum.Enum):
    gcn = GCN
    graphsage = GraphSAGE
    gat = GAT
    gin = GIN
    pna = PNA

class BasicGNNModule(ModuleWrapper):
    """ A module wrapper for a GAT network """
    def __init__(
        self,
        model_name: str,
        gnn_type: str,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        dropout_prob: float=0.1,
        dropout_bool: bool=True,
        device : torch.device=None
    ):
        super().__init__(model_name + f"_{gnn_type}", device=device)
        self.gnn_type = gnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.dropout = dropout_bool

        self.layer_name_list = []

        self.layer_name_list.append(f"{gnn_type}_model_")
        self.model = GNNLoader[gnn_type].value(
            in_channels=self.input_size,
            hidden_channels=self.hidden_size,
            num_layers=self.num_layers,
            out_channels=self.output_size,
            dropout= 0.0 if not self.dropout else self.dropout_prob,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the model """
        return self.model(x=x, edge_index=edge_index)
