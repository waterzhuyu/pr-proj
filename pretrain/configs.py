def get_base_config():
    """Base ViT config ViT."""
    return dict(
        dim=768,
        ff_dim=3072,
        num_heads=12,
        num_layers=12,
        dropout_rate=0.1,
        representation_size=768
    )


def get_b16_config():
    config = get_base_config()
    config.update(dict(patches=(16, 16)))
    return config


def get_l16_config():
    config = get_base_config()
    config.update(dict(
        patches=(16, 16),
        dim=1024,
        ff_dim=4096,
        num_heads=16,
        num_layers=24,
        attention_dropout_rate=0.0,
        dropout_rate=0.1,
        representaion_size=1024
    ))
    return config


PRETRAINED_MODELS = {
    'B_16': {
        'config': get_b16_config(),
        'num_classes': 21843,
        'image_size': (224, 224)
    },
    'L_16': {
        'config': get_l16_config(),
        'num_classes': 21843,
        'image_size': (224, 224)
    }
}
