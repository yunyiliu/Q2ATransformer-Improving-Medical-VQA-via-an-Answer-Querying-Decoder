from .Swin_Transformer.models import build_model
from .Swin_Transformer.config import get_config
import argparse
import torch


def parse_option(yaml_file, checkpoint_file, use_checkpoint):
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)

    args, unparsed = parser.parse_known_args([])
    args.cfg = yaml_file
    args.resume = checkpoint_file
    args.use_checkpoint = use_checkpoint
    config = get_config(args)

    return args, config


def from_pretrained(yaml_file, checkpoint_file, use_checkpoint=False):
    """

    Args:
        yaml_file:
        checkpoint_file:
        use_checkpoint:

    Returns:

    """
    _, config = parse_option(yaml_file, checkpoint_file, use_checkpoint)
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    model = build_model(config)
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f"Resume checkpoints from {checkpoint_file}")

    return model



