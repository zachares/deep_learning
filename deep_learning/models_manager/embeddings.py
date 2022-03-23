import torch
import torch.nn as nn

from deep_learning.models_manager.model_wrappers import ModuleWrapper

class Embedding(ModuleWrapper):
    """ A ModuleWrapper Class with a model composed of an embedding layer

        Additional Attributes:
            num_embed : the number of categories that require an embedding
            vector
            embed_dim : the number of dimensions an embedding vector
            should have

        NOTE: Please see ModuleWrapper class for other built-in methods.
    """
    def __init__(
        self,
        model_name : str,
        num_embed : int,
        embed_dim : int,
        padding_idx : int=None,
        max_norm : float=None,
        norm_type : float=None,
        scale_grad_by_freq : bool=False,
        sparse : bool=False,
        device : torch.device=None
    ):
        """ Inits an Embedding instance
            Please see https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
            for a description of the arguments to the init function
        """
        super().__init__(model_name + "_embed", device=device)
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.model = nn.Embedding(
            self.num_embed,
            self.embed_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse
        )

    def forward(self, idxs : torch.Tensor) -> torch.Tensor:
        """ Forward pass through the model """
        return self.model(idxs)
