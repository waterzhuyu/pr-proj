"""@{source} https://github.com/lukemelas/PyTorch-Pretrained-ViT"""
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

from .transformer import TransformerEncoder
from .utils import *
from .configs import PRETRAINED_MODELS


class PositionalEmbedding1D(nn.Module):
    """Add positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super(PositionalEmbedding1D, self).__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        """input: (batch_size, seq_len, emb_dim)"""
        return x + self.pos_embedding


class ViT(nn.Module):

    def __init__(
            self,
            name: str = None,
            pretrained: bool = False,
            finetune_all: bool = True,
            patches: int = 16,
            dim: int = 768,
            ff_dim: int = 3062,  # feed forward layer size
            num_heads: int = 12,
            num_layers: int = 12,
            dropout_rate: float = 0.1,
            representation_size: Optional[int] = None,  # when the model serve as feature extractor.
            load_repr_layer: bool = False,
            in_channels: int = 3,
            image_size: Optional[int] = None,
            num_classes: Optional[int] = None,
    ):
        super(ViT, self).__init__()
        if pretrained:
            config = PRETRAINED_MODELS[name]['config']
            patches = config['patches']
            dim = config['dim']
            ff_dim = config['ff_dim']
            num_heads = config['num_heads']
            num_layers = config['num_layers']
            dropout_rate = config['dropout_rate']
            representation_size = config['representation_size']

        if image_size is None:
            image_size = PRETRAINED_MODELS[name]['image_size']
        if num_classes is None:
            num_classes = PRETRAINED_MODELS[name]['num_classes']
        self.image_size = image_size

        h = image_size
        w = image_size
        fh, fw = patches, patches
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw

        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))

        # add class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        seq_len += 1

        self.pos_embedding = PositionalEmbedding1D(seq_len, dim)

        self.transformer = TransformerEncoder(num_layers, dim, num_heads, ff_dim, dropout_rate)

        # representation layer
        if representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(dim, representation_size)
            pre_logits_size = representation_size
        else:
            pre_logits_size = dim

        # classifier head
        self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
        self.fc = nn.Linear(pre_logits_size, num_classes)

        # Initialize weights
        self.init_weights()

        # load pretrained model
        if pretrained:
            pretrained_num_channels = 3
            pretrained_num_classes = PRETRAINED_MODELS[name]['num_classes']
            pretrained_image_size = PRETRAINED_MODELS[name]['image_size']

            load_pretrained_weights(
                self,
                weights_path=f'jax2pytorch/weights/{name}.pth',
                load_first_conv=(in_channels == pretrained_num_channels),
                load_fc=(num_classes == pretrained_num_classes),
                load_repr_layer=load_repr_layer,
                resize_positional_embedding=(image_size != pretrained_image_size)
            )
        if not finetune_all:
            rm_grad = [self.patch_embedding, self.pos_embedding, self.transformer]
            for layer in rm_grad:
                for param in layer.parameters():
                    param.requires_grad = False

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.normal_(self.pos_embedding.pos_embedding, std=0.02)
        nn.init.constant_(self.class_token, 0)

    def forward(self, x):
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b, d, gh, gw
        x = x.flatten(2).transpose(1, 2)  # b, gh*gw, d

        # cat class token
        x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)

        # add positional embedding
        x = self.pos_embedding(x)

        x = self.transformer(x)
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(self, 'fc'):
            x = self.norm(x)[:, 0]
            x = self.fc(x)
        return x
