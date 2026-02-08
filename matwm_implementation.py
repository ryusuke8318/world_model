"""
MATWM (Multi-Agent Transformer World Model) Implementation for Simple Tag

This implementation is based on the paper:
"Transformer World Model for Sample Efficient Multi-Agent Reinforcement Learning"
by Deihim et al. (2025), arXiv:2506.18537

Key Features:
1. Categorical VAE for latent representation
2. Transformer-based dynamics model
3. Teammate Predictor for social world modeling
4. Prioritized Replay Buffer
5. Imagination-based training
"""

import os
import math
import time
import random
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MATWMConfig:
    """
    Configuration for MATWM based on Table C.6 from the paper:
    "Transformer World Model for Sample Efficient Multi-Agent Reinforcement Learning"
    (arXiv:2506.18537)
    
    Settings optimized for simple_tag environment (4 agents).
    Paper recommends: 4-6 agents → batch_size=768, imagination_horizon=12
    """
    
    # Environment (simple_tag specific)
    max_cycles: int = 25
    num_agents: int = 4
    obs_dims: Dict[str, int] = field(default_factory=lambda: {
        'adversary_0': 16,
        'adversary_1': 16,
        'adversary_2': 16,
        'agent_0': 14,
    })
    # Unified observation dimension (max across all agents, for zero-padding)
    max_obs_dim: int = 16  # Max of all obs_dims values
    action_dim: int = 5  # 0-4: no-op, left, right, down, up
    
    # World Model Architecture (Table C.6)
    latent_dim: int = 32  # Latent dimension size
    num_classes: int = 32  # Number of categories per latent
    hidden_dim: int = 512  # Hidden dimension
    num_layers: int = 2  # Number of layers (paper uses 2, not 4)
    num_heads: int = 8  # Number of attention heads
    
    # Encoder Architecture (Table C.6)
    encoder_hidden_dim: int = 512  # MLP Encoder Hidden dim
    encoder_hidden_layers: int = 3  # MLP Encoder Hidden layers
    
    # Teammate Predictor (not specified in paper, using reasonable defaults)
    teammate_hidden_dim: int = 256
    
    # Agent Architecture (not specified in paper, using reasonable defaults)
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    
    # Training Parameters (Table C.6 + Appendix C adjustments)
    # Appendix C: "For environments with four to six agents and for 2c_vs_64zg, 
    # we use a batch size of 768 with an imagination horizon of 12."
    # simple_tag has 4 agents → use 4-6 agent settings
    wm_batch_size: int = 16  # World model train batch size (16 sequences × 64 steps = 1,024 samples)
    wm_batch_length: int = 64  # World model train batch length (sequence length)
    agent_batch_size: int = 768  # Agent train batch size (4-6 agents setting, Appendix C)
    sequence_length: int = 64  # Max sequence length
    imagination_horizon: int = 12  # Imagination horizon (4-6 agents: 12, Appendix C)
    imagination_context_length: int = 8  # Context for starting imagination rollout
    
    # Learning Rates (Table C.6)
    wm_learning_rate: float = 3e-5  # World Model Learning rate
    agent_learning_rate: float = 3e-4  # Actor+Critic Learning rate
    
    # Gradient Clipping (Table C.6)
    gradient_clip_wm: float = 1000.0  # Gradient clipping world model
    gradient_clip_agent: float = 10.0  # ★ FIX: 100→10 (Critic スパイク防止)
    
    # RL Parameters
    gamma: float = 0.99  # Discount factor
    lambda_gae: float = 0.95  # GAE lambda
    entropy_coef: float = 0.01  # ★ NEW: エントロピーボーナス係数（探索促進）
    
    # Replay Buffer (Table C.6)
    buffer_size: int = 50000  # Replay buffer size (will be adjusted in __post_init__)
    warmup_steps: int = 1000  # Random action steps before training (paper default)
    priority_decay: float = 0.9998  # Replay sampling priority decay (paper value)
    # Note: 0.9998 is very gradual. For faster adaptation: 0.995-0.997
    
    # Loss Weights (Table C.6)
    kl_weight: float = 0.5  # KL loss weight (β₁)
    representation_weight: float = 0.1  # Representation loss weight (β₂)
    free_nats: float = 1.0  # Free bits for KL loss
    
    # Additional Loss Weights (not in paper, using defaults)
    recon_weight: float = 1.0
    dynamics_weight: float = 1.0
    reward_weight: float = 1.0
    continuation_weight: float = 1.0
    teammate_weight: float = 0.5
    
    # Training Schedule (Table C.6)
    train_wm_every: int = 1  # Train world model every n steps
    train_agent_every: int = 1  # Train agent every n steps
    
    # Logging
    log_interval: int = 100
    save_interval: int = 5000
    eval_interval: int = 1000
    
    # Total Training (paper uses 50K for simple environments)
    total_steps: int = 50000  # 50K steps as per paper's recommendation
    
    def __post_init__(self):
        """Adjust buffer_size to be max(total_steps, 50000)"""
        self.buffer_size = max(self.total_steps, 50000)


# ============================================================================
# Utility Functions
# ============================================================================

def symlog(x):
    """Symmetric log transformation (Dreamer V3)"""
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    """Inverse of symlog"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot_encode(x, num_bins=255, low=-20, high=20):
    """Encode scalar values into two-hot representation"""
    x = torch.clamp(x, low, high)
    x_normalized = (x - low) / (high - low) * (num_bins - 1)
    lower = x_normalized.floor().long()
    upper = lower + 1
    upper = torch.clamp(upper, 0, num_bins - 1)
    lower = torch.clamp(lower, 0, num_bins - 1)
    
    weight_upper = x_normalized - lower.float()
    weight_lower = 1 - weight_upper
    
    encoding = torch.zeros(*x.shape, num_bins, device=x.device)
    encoding.scatter_add_(-1, lower.unsqueeze(-1), weight_lower.unsqueeze(-1))
    encoding.scatter_add_(-1, upper.unsqueeze(-1), weight_upper.unsqueeze(-1))
    
    return encoding


def two_hot_decode(encoding, num_bins=255, low=-20, high=20):
    """Decode two-hot representation back to scalar"""
    bins = torch.arange(num_bins, device=encoding.device, dtype=encoding.dtype)
    x_normalized = (encoding * bins).sum(dim=-1)
    x = x_normalized / (num_bins - 1) * (high - low) + low
    return x


def scale_action(action, agent_idx, action_dim=5):
    """Scale action to make it unique per agent"""
    return action + agent_idx * action_dim


def unscale_action(scaled_action, action_dim=5):
    """Unscale action to get original action and agent index"""
    agent_idx = scaled_action // action_dim
    action = scaled_action % action_dim
    return action, agent_idx


def pad_observation(obs, target_dim):
    """
    Zero-pad observation to target dimension.
    
    Args:
        obs: Observation array/tensor (shape: [..., obs_dim])
        target_dim: Target dimension after padding
    
    Returns:
        Padded observation (shape: [..., target_dim])
    """
    if isinstance(obs, np.ndarray):
        current_dim = obs.shape[-1]
        if current_dim >= target_dim:
            return obs[..., :target_dim]  # Truncate if larger
        pad_width = [(0, 0)] * (obs.ndim - 1) + [(0, target_dim - current_dim)]
        return np.pad(obs, pad_width, mode='constant', constant_values=0)
    elif isinstance(obs, torch.Tensor):
        current_dim = obs.shape[-1]
        if current_dim >= target_dim:
            return obs[..., :target_dim]  # Truncate if larger
        pad_size = target_dim - current_dim
        # pad format: (left, right, top, bottom, ...) - last dimension first
        return F.pad(obs, (0, pad_size), mode='constant', value=0)
    else:
        raise TypeError(f"Unsupported type: {type(obs)}")


# ============================================================================
# Prioritized Replay Buffer
# ============================================================================

class PrioritizedReplayBuffer:
    """Prioritized Replay Buffer that emphasizes recent experiences"""
    
    def __init__(self, capacity, priority_decay=0.995):
        self.capacity = capacity
        self.priority_decay = priority_decay
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, experience):
        """Add experience with highest priority (most recent)"""
        self.buffer.append(experience)
        self.priorities.append(1.0)  # Highest priority for new experience
        self.position = (self.position + 1) % self.capacity
        
        # Decay all priorities
        self.priorities = deque([p * self.priority_decay for p in self.priorities], 
                                maxlen=self.capacity)
    
    def sample(self, batch_size, sequence_length):
        """Sample batch of sequences with prioritization"""
        if len(self.buffer) < sequence_length:
            return None
        
        # Calculate sampling probabilities
        priorities = np.array(list(self.priorities))
        probs = priorities / priorities.sum()
        
        # Sample start indices
        max_start_idx = len(self.buffer) - sequence_length
        probs = probs[:max_start_idx + 1]
        probs = probs / probs.sum()
        
        start_indices = np.random.choice(
            max_start_idx + 1, 
            size=min(batch_size, max_start_idx + 1), 
            replace=False, 
            p=probs
        )
        
        # Extract sequences
        sequences = []
        for start_idx in start_indices:
            seq = [self.buffer[start_idx + i] for i in range(sequence_length)]
            sequences.append(seq)
        
        return sequences
    
    def sample_random(self, batch_size, sequence_length):
        """Sample batch of sequences uniformly (for agent training)"""
        if len(self.buffer) < sequence_length:
            return None
        
        max_start_idx = len(self.buffer) - sequence_length
        start_indices = np.random.choice(
            max_start_idx + 1, 
            size=min(batch_size, max_start_idx + 1), 
            replace=False
        )
        
        sequences = []
        for start_idx in start_indices:
            seq = [self.buffer[start_idx + i] for i in range(sequence_length)]
            sequences.append(seq)
        
        return sequences
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# World Model Components
# ============================================================================

class Encoder(nn.Module):
    """Encode observation to categorical latent distribution (Table C.6: 3 hidden layers, 512 dim)"""
    
    def __init__(self, obs_dim, latent_dim=32, num_classes=32, hidden_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Paper: MLP Encoder with 3 hidden layers of 512 dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * num_classes)
        )
    
    def forward(self, obs):
        """Return logits for categorical distribution"""
        logits = self.net(obs)
        logits = logits.reshape(*obs.shape[:-1], self.latent_dim, self.num_classes)
        return logits


class Decoder(nn.Module):
    """Decode latent state to observation"""
    
    def __init__(self, latent_dim=32, num_classes=32, obs_dim=16, hidden_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim * num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
    
    def forward(self, z):
        """Reconstruct observation from latent (one-hot)"""
        z_flat = z.reshape(*z.shape[:-2], -1)
        obs_recon = self.net(z_flat)
        return obs_recon


class ActionMixer(nn.Module):
    """Mix latent state with action"""
    
    def __init__(self, latent_dim=32, num_classes=32, action_dim=5, num_agents=4, hidden_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        # Action space is scaled: 0-4, 5-9, 10-14, 15-19 for 4 agents
        self.action_embed = nn.Embedding(action_dim * num_agents, hidden_dim)
        self.net = nn.Linear(latent_dim * num_classes + hidden_dim, hidden_dim)
    
    def forward(self, z, action):
        """Mix latent (one-hot) with action embedding"""
        z_flat = z.reshape(*z.shape[:-2], -1)
        action_emb = self.action_embed(action)
        mixed = torch.cat([z_flat, action_emb], dim=-1)
        return self.net(mixed)


class DynamicsModel(nn.Module):
    """Transformer-based dynamics model"""
    
    def __init__(self, latent_dim=32, num_classes=32, action_dim=5, num_agents=4,
                 hidden_dim=512, num_layers=4, num_heads=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.action_mixer = ActionMixer(latent_dim, num_classes, action_dim, num_agents, hidden_dim)
        
        # Vanilla Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Dynamics predictor
        self.dynamics_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * num_classes)
        )
    
    def forward(self, z_seq, action_seq):
        """
        Args:
            z_seq: (batch, seq_len, latent_dim, num_classes) - one-hot latent
            action_seq: (batch, seq_len) - scaled actions
        Returns:
            z_next_logits: (batch, seq_len, latent_dim, num_classes)
        """
        # Mix latent with action
        mixed = self.action_mixer(z_seq, action_seq)  # (batch, seq_len, hidden_dim)
        
        # Transformer
        h = self.transformer(mixed)  # (batch, seq_len, hidden_dim)
        
        # Predict next latent
        z_next_logits = self.dynamics_head(h)
        z_next_logits = z_next_logits.reshape(*h.shape[:-1], self.latent_dim, self.num_classes)
        
        return z_next_logits


class RewardPredictor(nn.Module):
    """Predict reward from latent state (two-hot symlog encoding)"""
    
    def __init__(self, latent_dim=32, num_classes=32, hidden_dim=256, num_bins=255):
        super().__init__()
        self.num_bins = num_bins
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim * num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_bins)
        )
    
    def forward(self, z):
        """Return two-hot distribution over reward bins"""
        z_flat = z.reshape(*z.shape[:-2], -1)
        logits = self.net(z_flat)
        return logits


class ContinuationPredictor(nn.Module):
    """Predict episode continuation (Bernoulli)"""
    
    def __init__(self, latent_dim=32, num_classes=32, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim * num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z):
        """Return logit for continuation probability"""
        z_flat = z.reshape(*z.shape[:-2], -1)
        logit = self.net(z_flat)
        return logit.squeeze(-1)


class TeammatePredictor(nn.Module):
    """
    Predict other agents' actions from focal agent's latent state
    
    ★ This is the CORE component for social world modeling ★
    
    It enables the focal agent to anticipate behaviors of other agents,
    which is crucial for coordination and competition.
    """
    
    def __init__(self, latent_dim=32, num_classes=32, action_dim=5, 
                 num_agents=4, hidden_dim=256):
        super().__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim
        
        # Separate predictor for each other agent
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim * num_classes, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)  # Unscaled action space
            )
            for _ in range(num_agents - 1)  # Exclude focal agent
        ])
    
    def forward(self, z, focal_agent_idx):
        """
        Predict actions of all other agents
        
        Args:
            z: (batch, seq_len, latent_dim, num_classes) - focal agent's latent
            focal_agent_idx: int - index of focal agent
        
        Returns:
            teammate_action_logits: dict mapping agent_idx -> logits
        """
        z_flat = z.reshape(*z.shape[:-2], -1)
        
        teammate_logits = {}
        predictor_idx = 0
        for agent_idx in range(self.num_agents):
            if agent_idx != focal_agent_idx:
                logits = self.predictors[predictor_idx](z_flat)
                teammate_logits[agent_idx] = logits
                predictor_idx += 1
        
        return teammate_logits


class WorldModel(nn.Module):
    """Complete MATWM World Model"""
    
    def __init__(self, config, agent_name):
        super().__init__()
        self.config = config
        self.agent_name = agent_name
        # Use unified max_obs_dim for all agents (with zero-padding)
        obs_dim = config.max_obs_dim
        
        self.encoder = Encoder(
            obs_dim, config.latent_dim, config.num_classes, config.hidden_dim
        )
        self.decoder = Decoder(
            config.latent_dim, config.num_classes, obs_dim, config.hidden_dim
        )
        self.dynamics = DynamicsModel(
            config.latent_dim, config.num_classes, config.action_dim, config.num_agents,
            config.hidden_dim, config.num_layers, config.num_heads
        )
        self.reward_predictor = RewardPredictor(
            config.latent_dim, config.num_classes, config.hidden_dim // 2
        )
        self.continuation_predictor = ContinuationPredictor(
            config.latent_dim, config.num_classes, config.hidden_dim // 2
        )
        self.teammate_predictor = TeammatePredictor(
            config.latent_dim, config.num_classes, config.action_dim,
            config.num_agents, config.teammate_hidden_dim
        )
    
    def encode(self, obs):
        """Encode observation to latent distribution"""
        logits = self.encoder(obs)
        # Sample with straight-through gradient
        z_one_hot = F.gumbel_softmax(logits, tau=1.0, hard=True)
        return z_one_hot, logits
    
    def decode(self, z):
        """Decode latent to observation"""
        return self.decoder(z)
    
    def predict_next(self, z_seq, action_seq):
        """Predict next latent state"""
        z_next_logits = self.dynamics(z_seq, action_seq)
        z_next = F.gumbel_softmax(z_next_logits, tau=1.0, hard=True)
        return z_next, z_next_logits
    
    def predict_reward(self, z):
        """Predict reward distribution"""
        return self.reward_predictor(z)
    
    def predict_continuation(self, z):
        """Predict continuation probability"""
        return self.continuation_predictor(z)
    
    def predict_teammates(self, z, focal_agent_idx):
        """Predict other agents' actions"""
        return self.teammate_predictor(z, focal_agent_idx)


# ============================================================================
# Agent Components
# ============================================================================

class Actor(nn.Module):
    """Policy network"""
    
    def __init__(self, latent_dim=32, num_classes=32, action_dim=5, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim * num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, z):
        """Return action logits"""
        z_flat = z.reshape(*z.shape[:-2], -1)
        return self.net(z_flat)


class Critic(nn.Module):
    """Value network (semi-centralized)"""
    
    def __init__(self, latent_dim=32, num_classes=32, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim * num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z):
        """Return value estimate"""
        z_flat = z.reshape(*z.shape[:-2], -1)
        return self.net(z_flat).squeeze(-1)


print("MATWM implementation loaded successfully!")
print("Key components:")
print("  - Encoder/Decoder (Categorical VAE)")
print("  - DynamicsModel (Transformer-based)")
print("  - TeammatePredictor ★ Social World Model Core ★")
print("  - Actor/Critic Networks")
print("  - PrioritizedReplayBuffer")


