import os
import torch
import numpy

# Autorise les types NumPy pour le chargement sécurisé (nécessaire pour PyTorch 2.6+)
torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])

def save_model(agent, algo, seed, filepath=None):
    os.makedirs("models", exist_ok=True)
    if filepath is None:
        filepath = f"models/{algo}_seed{seed}.pth"
        
    if algo.startswith("dqn"):
        torch.save(agent.q_net.state_dict(), filepath)
        print(f"  → Modèle DQN sauvegardé dans {filepath}")
    elif algo.startswith("reinforce"):
        torch.save(agent.policy.state_dict(), filepath)
        print(f"  → Modèle REINFORCE sauvegardé dans {filepath}")
    elif algo.startswith("q_learning"):
        # Convert defaultdict to standard dict for pickling
        torch.save(dict(agent.q_table), filepath)
        print(f"  → Table Q-Learning sauvegardée dans {filepath}")
    else:
        # random or heuristic: save a dummy file to allow eval.py to find it
        torch.save({"type": algo, "status": "baseline"}, filepath)
        print(f"  → Baseline {algo.upper()} 'sauvegardée' dans {filepath}")

def load_model(agent, algo, filepath):
    if not os.path.exists(filepath):
        print(f"[ERREUR] Le modèle {filepath} n'existe pas.")
        return False
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if algo.startswith("dqn"):
        # Utilisation de weights_only=True par défaut (standard pour state_dict)
        agent.q_net.load_state_dict(torch.load(filepath, map_location=device, weights_only=True))
        agent.target_net.load_state_dict(agent.q_net.state_dict())
        agent.epsilon = 0.0 # Greedy mode
        print(f"  → Modèle DQN chargé depuis {filepath}")
        
    elif algo.startswith("reinforce"):
        agent.policy.load_state_dict(torch.load(filepath, map_location=device, weights_only=True))
        print(f"  → Modèle REINFORCE chargé depuis {filepath}")
        
    elif algo.startswith("q_learning"):
        # Pour le Q-Learning, on charge des dictionnaires bruts (potentiellement avec des types NumPy)
        # On utilise weights_only=False si le add_safe_globals ne suffisait pas pour des clés complexes
        loaded_dict = torch.load(filepath, map_location="cpu", weights_only=False)
        for k, v in loaded_dict.items():
            agent.q_table[k] = v
        agent.epsilon = 0.0 # Greedy mode
        print(f"  → Table Q-Learning chargée depuis {filepath}")
    
    else:
        # random or heuristic
        print(f"  → Baseline {algo.upper()} prête (aucune pondération à charger)")
    
    return True
