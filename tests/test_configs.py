import pytest
import yaml
import glob
import os

def test_load_all_configs():
    config_files = glob.glob("configs/*.yaml")
    assert len(config_files) > 0, "Aucun fichier de config trouvé dans configs/"
    
    mandatory_fields = ["env", "algo", "hyperparameters"]
    
    for cf in config_files:
        with open(cf, 'r') as f:
            config = yaml.safe_load(f)
            assert config is not None, f"Le fichier {cf} est vide ou invalide."
            for field in mandatory_fields:
                assert field in config, f"Le champ obligatoire '{field}' est manquant dans {cf}"
            
            # Vérifier qu'il y a des hyperparamètres (sauf pour les baselines simples)
            if config["algo"] not in ["heuristic", "random"]:
                assert len(config["hyperparameters"]) > 0, f"Aucun hyperparamètre défini dans {cf}"
