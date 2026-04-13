import os
import torch
from box import Box
import yaml
import numpy as np
from src.data.dataset import UniRigDatasetModule, DatasetConfig
from src.data.transform import TransformConfig
from src.tokenizer.spec import TokenizerConfig
from src.tokenizer.parse import get_tokenizer
from src.model.parse import get_model
from src.data.asset import Asset
from src.data.raw_data import RawData

def test_dataloader():
    # Load model config to initialize model and get process_fn
    with open('configs/model/unirig_rignet.yaml', 'r') as f:
        model_config = Box(yaml.safe_load(f))
    
    # We need a tokenizer to initialize the model
    with open('configs/tokenizer/tokenizer_rignet.yaml', 'r') as f:
        tokenizer_config_raw = Box(yaml.safe_load(f))
        tokenizer_config = TokenizerConfig.parse(tokenizer_config_raw)
    
    tokenizer = get_tokenizer(tokenizer_config)
    model = get_model(**model_config, tokenizer=tokenizer)

    # Create a temporary config for testing
    data_config = {
        'validate_dataset_config': {
            'shuffle': False,
            'batch_size': 1,
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False,
            'datapath_config': {
                'input_dataset_dir': './dataset_clean/rigxl',
                'use_prob': False,
                'data_path': {
                    'vroid': [
                        ['./dataset_clean/datalist/vroid_test.txt', 1.0],
                    ]
                }
            }
        }
    }
    data_config = Box(data_config)
    
    # Manually modify transform config to set copy_joint_to_tail to True for testing
    with open('configs/transform/inference_ar_transform.yaml', 'r') as f:
        transform_config_raw = Box(yaml.safe_load(f))
        
    transform_config_dict = transform_config_raw.validate_transform_config
    transform_config_dict.tail_config.copy_joint_to_tail = True
    
    transform_config = TransformConfig.parse(transform_config_dict)

    validate_dataset_config = DatasetConfig.parse(data_config.validate_dataset_config).split_by_cls()

    dataset = UniRigDatasetModule(
        process_fn=model._process_fn,
        validate_dataset_config=validate_dataset_config,
        validate_transform_config=transform_config,
        tokenizer_config=tokenizer_config,
    )
    
    dataset.setup()
    val_loader = dataset.val_dataloader()
    
    print(f"Validation loader type: {type(val_loader)}")
    if isinstance(val_loader, dict):
        for cls, loader in val_loader.items():
            print(f"Class: {cls}, number of batches: {len(loader)}")
            for batch in loader:
                print(f"Batch keys: {batch.keys()}")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: {v.shape}, {v.dtype}")
                    elif isinstance(v, list):
                        if len(v) > 0:
                            print(f"  {k}: list of length {len(v)}, first element type {type(v[0])}")
                        else:
                            print(f"  {k}: empty list")
                    else:
                        print(f"  {k}: {type(v)}")
                break # Only one batch
    else:
        print(f"Number of batches: {len(val_loader)}")
        for batch in val_loader:
            print(f"Batch keys: {batch.keys()}")
            break

if __name__ == "__main__":
    test_dataloader()
