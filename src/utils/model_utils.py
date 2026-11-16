import os
import torch
import numpy

# Allow NumPy types for safe loading (required for PyTorch 2.6+)
torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])

def save_model(agent, algo, seed, filepath=None):
    os.makedirs("models", exist_ok=True)
    if filepath is None:
        filepath = f"models/{algo}_seed{seed}.pth"
        
    if algo.startswith("dqn"):
        torch.save(agent.q_net.state_dict(), filepath)
        print(f"  → DQN model saved to {filepath}")
    elif algo.startswith("reinforce"):
        torch.save(agent.policy.state_dict(), filepath)
        print(f"  → REINFORCE model saved to {filepath}")
    elif algo.startswith("q_learning"):
        # Convert defaultdict to standard dict for pickling
        torch.save(dict(agent.q_table), filepath)
        print(f"  → Q-Learning table saved to {filepath}")
    else:
        raise ValueError(f"Nothing to save for algo '{algo}'")

def load_model(agent, algo, filepath):
    if not os.path.exists(filepath):
        print(f"[ERROR] Model {filepath} does not exist.")
        return False
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if algo.startswith("dqn"):
        # Use weights_only=True by default (standard for state_dict)
        agent.q_net.load_state_dict(torch.load(filepath, map_location=device, weights_only=True))
        agent.target_net.load_state_dict(agent.q_net.state_dict())
        agent.epsilon = 0.0 # Greedy mode
        print(f"  → DQN model loaded from {filepath}")
        
    elif algo.startswith("reinforce"):
        agent.policy.load_state_dict(torch.load(filepath, map_location=device, weights_only=True))
        print(f"  → REINFORCE model loaded from {filepath}")
        
    elif algo.startswith("q_learning"):
        # Q-Learning loads raw dicts (possibly containing NumPy types)
        # Use weights_only=False in case add_safe_globals is not enough for complex keys
        loaded_dict = torch.load(filepath, map_location="cpu", weights_only=False)
        for k, v in loaded_dict.items():
            agent.q_table[k] = v
        agent.epsilon = 0.0 # Greedy mode
        print(f"  → Q-Learning table loaded from {filepath}")
    else:
        raise ValueError(f"No model format for algo '{algo}'")

    return True
