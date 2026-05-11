import pytest
import torch
import torch.nn as nn
import numpy as np
import os
from unittest.mock import MagicMock
from src.utils.replay_buffer import ReplayBuffer
from src.utils.model_utils import save_model, load_model

def test_replay_buffer():
    capacity = 10
    device = torch.device("cpu")
    buffer = ReplayBuffer(capacity=capacity, device=device)
    
    # Remplir au delà de la capacité  
    for i in range(capacity + 5):
        buffer.push(np.random.rand(8), 1, 1.0, np.random.rand(8), False)
    
    assert len(buffer) == capacity
    
    # Tester le sample  
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=5)
    assert states.shape == (5, 8)
    assert actions.shape == (5,)
    assert rewards.shape == (5,)

def test_model_utils_dqn():
    # Mock DQNAgent with q_net  
    mock_agent = MagicMock()
    mock_agent.q_net = nn.Sequential(nn.Linear(8, 4))
    
    path = "tests/test_dqn_save.pth"
    
    # Sauvegarde  
    save_model(mock_agent, "dqn_test", 99, filepath=path)
    assert os.path.exists(path)
    
    # Chargement
    new_agent = MagicMock()
    new_agent.q_net = nn.Sequential(nn.Linear(8, 4))
    new_agent.target_net = nn.Sequential(nn.Linear(8, 4))
    
    success = load_model(new_agent, "dqn_test", path)
    assert success
    
    # Nettoyage
    if os.path.exists(path):
        os.remove(path)

def test_video_utils_path_exists():
    from src.utils.video_utils import concat_episode_videos
    assert callable(concat_episode_videos)
