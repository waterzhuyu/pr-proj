import numpy as np
import torch

from .configs import PRETRAINED_MODELS


def load_pretrained_weights(
        model,
        weights_path=None,
        load_first_conv=True,
        load_fc=True,
        load_repr_layer=False,
        resize_positional_embedding=False,
):
    """Loads pretrained weights by weight path"""
    state_dict = torch.load(weights_path)

    # remove some layers
    expected_missing_keys = []
    if not load_first_conv and 'patch_embedding.weight' in state_dict:
        expected_missing_keys += ['patch_embedding.weight', 'patch_embedding.bias']
    if not load_fc and 'fc.weight' in state_dict:
        expected_missing_keys += ['fc.weight', 'fc.bias']
    if not load_repr_layer and 'pre_logits.weight' in state_dict:
        expected_missing_keys += ['pre_logits.weight', 'pre_logits.bias']
    for key in expected_missing_keys:
        state_dict.pop(key)

    # change size of positional embeddings
    if resize_positional_embedding:
        pos_emb = state_dict['pos_embedding.pos_embedding']
        pos_emb_new = model.state_dict()['pos_embedding.pos_embedding']
        state_dict['pos_embedding.pos_embedding'] = resize_positional_embedding_(
            pos_emb, pos_emb_new, hasattr(model, 'cls_token'))

    # load state dict
    ret = model.load_state_dict(state_dict, strict=False)
    return ret


def resize_positional_embedding_(pos_emb, pos_emb_new, has_cls_token=True):
    """Rescale the grid of positional embedding in a sensible manner."""
    from scipy.ndimage import zoom

    ntok_new = pos_emb_new.shape[1]
    if has_cls_token:
        pos_emb_token, pos_emb_grid = pos_emb[:, :1], pos_emb[0, 1:]
        ntok_new -= 1
    else:
        pos_emb_token, pos_emb_grid = pos_emb[:, :0], pos_emb[0]

    # get old and new grid sizes
    gs_old = int(np.sqrt(len(pos_emb_grid)))
    gs_new = int(np.sqrt(ntok_new))
    pos_emb_grid = pos_emb_grid.reshape(gs_old, gs_old, -1)

    # rescale grid
    zoom_factor = (gs_new / gs_old, gs_new / gs_old, 1)
    pos_emb_grid = zoom(pos_emb_grid, zoom_factor, order=1)
    pos_emb_grid = pos_emb_grid.reshape(1, gs_new * gs_new, -1)
    pos_emb_grid = torch.from_numpy(pos_emb_grid)

    # Deal with class token and return
    pos_emb = torch.cat([pos_emb_token, pos_emb_grid], dim=1)
    return pos_emb
