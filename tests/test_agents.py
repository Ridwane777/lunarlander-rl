import pytest
import numpy as np
from unittest.mock import MagicMock
from src.agents.q_learning import QLearningAgent
from src.agents.reinforce_agent import ReinforceAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent

@pytest.fixture
def dummy_config():
    return {
        "gamma": 0.99,
        "lr": 0.001,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "batch_size": 32,
        "buffer_size": 1000,
        "target_update_freq": 10,
        "update_every": 1,
        "hidden": 64,
        "grad_clip": 1.0
    }

def test_random_agent():
    # Mock environment with action_space
    mock_env = MagicMock()
    mock_env.action_space.sample.return_value = 1
    
    agent = RandomAgent(env=mock_env)
    obs = np.random.rand(8)
    action = agent.select_action(obs)
    assert action == 1
    mock_env.action_space.sample.assert_called_once()

def test_heuristic_agent():
    agent = HeuristicAgent()
    # Fast descent test (should fire the main engine - action 2)
    obs = np.array([0, 0.4, 0, -2.0, 0, 0, 0, 0])
    action = agent.select_action(obs)
    assert action == 2

def test_q_learning_agent(dummy_config):
    agent = QLearningAgent(n_actions=4, config=dummy_config)
    obs = np.random.rand(8)
    
    # Test discretize
    state = agent.discretize(obs)
    assert isinstance(state, tuple)
    
    # Test action selection
    action = agent.select_action(obs)
    assert action in [0, 1, 2, 3]

def test_reinforce_agent(dummy_config):
    agent = ReinforceAgent(obs_dim=8, n_actions=4, config=dummy_config)
    obs = np.random.rand(8)
    
    # Test action selection
    action = agent.select_action(obs)
    assert action in [0, 1, 2, 3]
    assert len(agent.log_probs) == 1

def test_dqn_agent(dummy_config):
    agent = DQNAgent(obs_dim=8, n_actions=4, config=dummy_config)
    obs = np.random.rand(8)
    
    # Test action selection  
    action = agent.select_action(obs)
    assert action in [0, 1, 2, 3]
    
    # Test memory push (store)  
    agent.store(obs, action, 1.0, obs, False)
    assert len(agent.buffer) == 1
