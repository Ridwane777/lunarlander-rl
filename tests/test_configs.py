import pytest
import yaml
import glob
import os

def test_load_all_configs():
    config_files = glob.glob("configs/*.yaml")
    assert len(config_files) > 0, "No config file found in configs/"
    
    mandatory_fields = ["env", "algo", "hyperparameters"]
    
    for cf in config_files:
        with open(cf, 'r') as f:
            config = yaml.safe_load(f)
            assert config is not None, f"File {cf} is empty or invalid."
            for field in mandatory_fields:
                assert field in config, f"Required field '{field}' missing in {cf}"
            
            assert len(config["hyperparameters"]) > 0, f"No hyperparameters defined in {cf}"
