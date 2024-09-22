import os
import torch
from .model import TransformerModel
from .utils import load_pretrained
import json

def get_cell_embed(gene_encode_cfg):
    gene_encode_path = getattr(gene_encode_cfg, 'scgpt_path')
    is_absolute_path_exists = os.path.exists(gene_encode_path)
    if is_absolute_path_exists :
        model_config_file = gene_encode_path / "args.json"
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        model = TransformerModel(
            ntoken=10,
            d_model=model_configs["embsize"],
            nhead=model_configs["nheads"],
            d_hid=model_configs["d_hid"],
            nlayers=model_configs["nlayers"],
            nlayers_cls=model_configs["n_layers_cls"],
            n_cls=1,
            vocab=None,
            dropout=model_configs["dropout"],
            pad_token=model_configs["pad_token"],
            pad_value=model_configs["pad_value"],
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            explicit_zero_prob=False,
            use_fast_transformer=False,
            fast_transformer_backend="flash",
            pre_norm=False,
        )
        load_pretrained(model, torch.load(model_file, map_location=device), verbose=False)
        model.eval()
        return model

    raise ValueError(f'Unknown path: {gene_encode_path}')