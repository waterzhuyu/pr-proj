import json
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# import pretrain


npz_files = {
    'B_16': 'jax_weights/ViT-B_16.npz'
}


def jax2pytorch(k):
    """Convert key of jax npz to torch state dict keys"""
    k = k.replace('Transformer/encoder_norm', 'norm')
    # k = k.replace('Transformer', 'transformer_encoder')
    k = k.replace('LayerNorm_0', 'norm1')
    k = k.replace('LayerNorm_2', 'norm2')
    k = k.replace('MlpBlock_3/Dense_0', 'pwff.fc1')
    k = k.replace('MlpBlock_3/Dense_1', 'pwff.fc2')
    k = k.replace('MultiHeadDotProductAttention_1/out', 'proj')
    k = k.replace('MultiHeadDotProductAttention_1/query', 'attn.proj_q')
    k = k.replace('MultiHeadDotProductAttention_1/key', 'attn.proj_k')
    k = k.replace('MultiHeadDotProductAttention_1/value', 'attn.proj_v')
    k = k.replace('Transformer/posembed_input', 'pos_embedding')
    k = k.replace('encoderblock_', 'blks.')
    k = 'patch_embedding.bias' if k == 'embedding/bias' else k
    k = 'patch_embedding.weight' if k == 'embedding/kernel' else k
    k = 'class_token' if k == 'cls' else k
    k = k.replace('head', 'fc')
    k = k.replace('kernel', 'weight')
    k = k.replace('scale', 'weight')
    k = k.replace('/', '.')
    k = k.lower()
    return k


def convert(npz, state_dict):
    new_state_dict = {}
    pytorch_k2v = {jax2pytorch(k): v for k, v in npz.items()}
    for torch_k, torch_v in state_dict.items():
        # Naming
        if 'self_attn.out_proj.weight' in torch_k:
            v = pytorch_k2v[torch_k]
            v = v.reshape(v.shape[0] * v.shape[1], v.shape[2])
        elif 'self_attn.in_proj_' in torch_k:
            v = np.stack((pytorch_k2v[torch_k + '*q'],
                          pytorch_k2v[torch_k + '*k'],
                          pytorch_k2v[torch_k, '*v']), axis=0)
        else:
            if torch_k not in pytorch_k2v:
                print(torch_k, list(pytorch_k2v.keys()))
                assert False
            v = pytorch_k2v[torch_k]
        v = torch.from_numpy(v)

        if '.weight' in torch_k:
            if len(torch_v.shape) == 2:
                v = v.transpose(0, 1)
            if len(torch_v.shape) == 4:
                v = v.permute(3, 2, 0, 1)
        if 'proj.weight' in torch_k:
            v = v.transpose(0, 1)
            v = v.reshape(-1, v.shape[-1])
        if 'attn.proj_' in torch_k and 'weight' in torch_k:
            v = v.permute(0, 2, 1)
            v = v.reshape(-1, v.shape[-1])
        if 'attn.proj_' in torch_k and 'bias' in torch_k:
            v = v.reshape(-1)
        new_state_dict[torch_k] = v
    return new_state_dict


npz_files = {
    'B_16': 'jax_weights/finetune/ViT-B_16.npz'
}


for name, filename in npz_files.items():
    npz = np.load(filename)
    for k, v in npz.items():
        print(k)
        print(v.shape)
    # model = pretrain.ViT(name=name, num_classes=1000, image_size=384, pretrained=False)
    # print(model)
    # new_state_dict = convert(npz, model.state_dict())
    #
    # model.load_state_dict(new_state_dict)
    #
    # new_filename = f'weights/finetune/{name}.pth'
    # torch.save(new_state_dict, new_filename, _use_new_zipfile_serialization=False)
