"""
MATWM Agent Implementation

Complete agent implementation with training loops for:
1. World Model training (with prioritized replay)
2. Actor-Critic training (with imagination)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

from matwm_implementation import (
    WorldModel, Actor, Critic,
    PrioritizedReplayBuffer,
    symlog, symexp, two_hot_encode, two_hot_decode,
    scale_action, pad_observation
)


class MATWMAgent:
    """Complete MATWM Agent with World Model and Actor-Critic"""
    
    # エージェント名→インデックスの静的マッピング（simple_tag_v3 用）
    # store_experience では other_actions が {agent_name: action} 形式で保存されるが、
    # TeammatePredictor は {agent_idx: logits} で返すため、変換が必要。
    AGENT_NAME_TO_IDX = {
        'adversary_0': 0,
        'adversary_1': 1,
        'adversary_2': 2,
        'agent_0': 3,
    }
    
    def __init__(self, config, agent_name, agent_idx, device, shared_world_model=None):
        self.config = config
        self.agent_name = agent_name
        self.agent_idx = agent_idx
        self.device = device
        
        # World Model (shared across all agents if provided)
        if shared_world_model is not None:
            self.world_model = shared_world_model
            self.owns_world_model = False
        else:
            self.world_model = WorldModel(config, agent_name).to(device)
            self.owns_world_model = True
        
        # Actor-Critic (always individual per agent)
        self.actor = Actor(
            config.latent_dim, config.num_classes, 
            config.action_dim, config.actor_hidden_dim
        ).to(device)
        self.critic = Critic(
            config.latent_dim, config.num_classes, config.critic_hidden_dim
        ).to(device)
        
        # Optimizers (paper: WM=3e-5, Actor+Critic=3e-4)
        # World Model optimizer: only create if this agent owns the world model
        if self.owns_world_model:
            self.wm_optimizer = torch.optim.Adam(
                self.world_model.parameters(), lr=config.wm_learning_rate
            )
        else:
            self.wm_optimizer = None  # Shared WM is optimized elsewhere
            
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.agent_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.agent_learning_rate
        )
        
        # Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            config.buffer_size, config.priority_decay
        )
        
        self.training = True
    
    def select_action(self, obs, deterministic=False):
        """Select action using actor (with zero-padding for observation)"""
        with torch.no_grad():
            # Zero-pad observation to max_obs_dim
            obs_padded = pad_observation(obs, self.config.max_obs_dim)
            obs_tensor = torch.FloatTensor(obs_padded).unsqueeze(0).to(self.device)
            z, _ = self.world_model.encode(obs_tensor)
            action_logits = self.actor(z)
            
            if deterministic:
                action = action_logits.argmax(dim=-1)
            else:
                action_dist = torch.distributions.Categorical(logits=action_logits)
                action = action_dist.sample()
            
            return action.item()
    
    def store_experience(self, obs, action, reward, next_obs, done, other_actions):
        """Store experience in replay buffer (with zero-padding for observation)"""
        # Zero-pad observations to max_obs_dim
        obs_padded = pad_observation(obs, self.config.max_obs_dim)
        next_obs_padded = pad_observation(next_obs, self.config.max_obs_dim)
        
        experience = {
            'obs': obs_padded,
            'action': action,
            'reward': reward,
            'next_obs': next_obs_padded,
            'done': done,
            'other_actions': other_actions  # Actions of other agents
        }
        self.replay_buffer.push(experience)
    
    def train_world_model(self):
        """
        Train world model on prioritized replay buffer.
        
        NOTE: In the paper's implementation (Algorithm 2, L28-29), world model training
        should sample from ALL agents' replay buffers and train once per step.
        This method samples only from this agent's buffer. The training loop should
        collect samples from all agents and call train_world_model_shared() instead.
        """
        sequences = self.replay_buffer.sample(
            self.config.wm_batch_size, self.config.wm_batch_length
        )
        if sequences is None:
            return {}
        
        return self._train_world_model_on_sequences(sequences)
    
    @staticmethod
    def create_shared_world_model(config, device):
        """
        Create a shared world model instance for all agents.
        Returns: (world_model, optimizer)
        """
        world_model = WorldModel(config, "shared").to(device)
        wm_optimizer = torch.optim.Adam(
            world_model.parameters(), 
            lr=config.wm_learning_rate
        )
        return world_model, wm_optimizer
    
    @staticmethod
    def train_world_model_shared(agents_dict, config, device, shared_wm_optimizer=None):
        """
        Train shared world model using samples from all agents' replay buffers.
        This is the correct implementation according to Algorithm 2 (L28-29).
        
        Args:
            agents_dict: Dictionary of {agent_name: MATWMAgent}
            config: MATWMConfig
            device: torch device
            shared_wm_optimizer: Optimizer for shared world model (if None, uses first agent's)
            
        Returns:
            dict: Training metrics
        """
        # Sample sequences from each agent's replay buffer
        sequences_per_agent = config.wm_batch_size // len(agents_dict)
        all_sequences = []
        agent_indices = []  # Track which agent each sequence belongs to
        
        for agent_idx, (agent_name, agent) in enumerate(agents_dict.items()):
            sequences = agent.replay_buffer.sample(
                sequences_per_agent,
                config.wm_batch_length
            )
            if sequences:
                all_sequences.extend(sequences)
                agent_indices.extend([agent_idx] * len(sequences))
        
        if len(all_sequences) != config.wm_batch_size:
            return {}  # Not enough data yet
        
        # Get shared world model and optimizer
        first_agent = list(agents_dict.values())[0]
        world_model = first_agent.world_model
        wm_optimizer = shared_wm_optimizer if shared_wm_optimizer is not None else first_agent.wm_optimizer
        
        if wm_optimizer is None:
            raise ValueError("No optimizer available for shared world model. Use create_shared_world_model() or ensure one agent owns the WM.")
        
        if len(all_sequences) != config.wm_batch_size:
            return {}  # Not enough data yet
        
        # Prepare mixed batch from all agents
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_obs_batch = []
        done_batch = []
        other_actions_batch = []
        agent_idx_batch = []
        
        for seq_idx, seq in enumerate(all_sequences):
            obs_seq = np.stack([exp['obs'] for exp in seq])
            action_seq = np.array([exp['action'] for exp in seq])
            reward_seq = np.array([exp['reward'] for exp in seq])
            next_obs_seq = np.stack([exp['next_obs'] for exp in seq])
            done_seq = np.array([exp['done'] for exp in seq])
            other_actions_seq = [exp['other_actions'] for exp in seq]
            
            obs_batch.append(obs_seq)
            action_batch.append(action_seq)
            reward_batch.append(reward_seq)
            next_obs_batch.append(next_obs_seq)
            done_batch.append(done_seq)
            other_actions_batch.append(other_actions_seq)
            agent_idx_batch.append(agent_indices[seq_idx])
        
        obs_batch = torch.FloatTensor(np.stack(obs_batch)).to(device)
        action_batch = torch.LongTensor(np.stack(action_batch)).to(device)
        reward_batch = torch.FloatTensor(np.stack(reward_batch)).to(device)
        next_obs_batch = torch.FloatTensor(np.stack(next_obs_batch)).to(device)
        done_batch = torch.FloatTensor(np.stack(done_batch)).to(device)
        agent_idx_batch = torch.LongTensor(agent_idx_batch).to(device)
        
        # Scale actions based on agent index
        scaled_action_batch = action_batch.clone()
        for seq_idx in range(len(all_sequences)):
            scaled_action_batch[seq_idx] += agent_idx_batch[seq_idx] * config.action_dim
        
        # Get any agent's world model (they are shared)
        first_agent = list(agents_dict.values())[0]
        world_model = first_agent.world_model
        wm_optimizer = first_agent.wm_optimizer
        
        # Encode observations
        z, z_logits = world_model.encode(obs_batch)
        z_next_target, z_next_logits_target = world_model.encode(next_obs_batch)
        
        # Reconstruction loss
        obs_recon = world_model.decode(z)
        recon_loss = F.mse_loss(obs_recon, obs_batch)
        
        # Dynamics loss
        z_next_pred, z_next_pred_logits = world_model.predict_next(z, scaled_action_batch)
        dynamics_loss = F.cross_entropy(
            z_next_pred_logits.reshape(-1, config.num_classes),
            z_next_target.reshape(-1, config.num_classes).argmax(dim=-1)
        )
        
        # Reward loss (two-hot symlog)
        reward_logits = world_model.predict_reward(z_next_pred)
        reward_symlog = symlog(reward_batch)
        reward_target = two_hot_encode(reward_symlog)
        reward_loss = F.cross_entropy(
            reward_logits.reshape(-1, 255),
            reward_target.reshape(-1, 255).argmax(dim=-1)
        )
        
        # Continuation loss
        cont_logits = world_model.predict_continuation(z_next_pred)
        cont_target = 1.0 - done_batch
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)
        
        # Teammate prediction loss
        # ★ FIX: other_actions は {agent_name(str): action} 形式で保存されているが、
        # predict_teammates は {agent_idx(int): logits} を返す。
        # agent_name → agent_idx への変換が必要。
        teammate_loss = 0.0
        count = 0
        
        for seq_idx in range(len(all_sequences)):
            focal_agent_idx = agent_idx_batch[seq_idx].item()
            teammate_logits_dict = world_model.predict_teammates(
                z[seq_idx:seq_idx+1], focal_agent_idx
            )
            
            for t in range(config.wm_batch_length):
                if other_actions_batch[seq_idx][t] is not None:
                    # agent_name → agent_idx に変換してからマッチング
                    other_actions_by_idx = {}
                    for aname, aact in other_actions_batch[seq_idx][t].items():
                        aidx = MATWMAgent.AGENT_NAME_TO_IDX.get(aname)
                        if aidx is not None:
                            other_actions_by_idx[aidx] = aact
                    
                    for other_agent_idx, logits in teammate_logits_dict.items():
                        other_action = other_actions_by_idx.get(other_agent_idx)
                        if other_action is not None:
                            target = torch.LongTensor([other_action]).to(device)
                            teammate_loss += F.cross_entropy(
                                logits[0, t, :].unsqueeze(0), target
                            )
                            count += 1
        
        if count > 0:
            teammate_loss = teammate_loss / count
        else:
            teammate_loss = torch.tensor(0.0).to(device)
        
        # KL divergence with free nats
        kl_loss = F.kl_div(
            F.log_softmax(z_logits.reshape(-1, config.num_classes), dim=-1),
            F.softmax(z_next_logits_target.reshape(-1, config.num_classes), dim=-1),
            reduction='batchmean'
        )
        kl_loss = torch.maximum(kl_loss, torch.tensor(config.free_nats).to(device))
        
        # Total loss
        total_loss = (
            config.recon_weight * recon_loss +
            config.dynamics_weight * dynamics_loss +
            config.reward_weight * reward_loss +
            config.continuation_weight * cont_loss +
            config.teammate_weight * teammate_loss +
            config.kl_weight * kl_loss
        )
        
        # Update shared world model once
        shared_wm_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(world_model.parameters(), config.gradient_clip_wm)
        shared_wm_optimizer.step()
        
        return {
            'wm_total_loss': total_loss.item(),
            'wm_recon_loss': recon_loss.item(),
            'wm_dynamics_loss': dynamics_loss.item(),
            'wm_reward_loss': reward_loss.item(),
            'wm_cont_loss': cont_loss.item(),
            'wm_teammate_loss': teammate_loss.item() if isinstance(teammate_loss, torch.Tensor) else teammate_loss,
            'wm_kl_loss': kl_loss.item(),
        }
    
    def _train_world_model_on_sequences(self, sequences):
        """Internal method to train world model on given sequences"""
        # Prepare batch
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_obs_batch = []
        done_batch = []
        other_actions_batch = []
        
        for seq in sequences:
            obs_seq = np.stack([exp['obs'] for exp in seq])
            action_seq = np.array([exp['action'] for exp in seq])
            reward_seq = np.array([exp['reward'] for exp in seq])
            next_obs_seq = np.stack([exp['next_obs'] for exp in seq])
            done_seq = np.array([exp['done'] for exp in seq])
            other_actions_seq = [exp['other_actions'] for exp in seq]
            
            obs_batch.append(obs_seq)
            action_batch.append(action_seq)
            reward_batch.append(reward_seq)
            next_obs_batch.append(next_obs_seq)
            done_batch.append(done_seq)
            other_actions_batch.append(other_actions_seq)
        
        obs_batch = torch.FloatTensor(np.stack(obs_batch)).to(self.device)
        action_batch = torch.LongTensor(np.stack(action_batch)).to(self.device)
        reward_batch = torch.FloatTensor(np.stack(reward_batch)).to(self.device)
        next_obs_batch = torch.FloatTensor(np.stack(next_obs_batch)).to(self.device)
        done_batch = torch.FloatTensor(np.stack(done_batch)).to(self.device)
        
        # Scale actions
        scaled_action_batch = action_batch + self.agent_idx * self.config.action_dim
        
        # Encode observations
        z, z_logits = self.world_model.encode(obs_batch)
        z_next_target, z_next_logits_target = self.world_model.encode(next_obs_batch)
        
        # Reconstruction loss
        obs_recon = self.world_model.decode(z)
        recon_loss = F.mse_loss(obs_recon, obs_batch)
        
        # Dynamics loss
        z_next_pred, z_next_pred_logits = self.world_model.predict_next(z, scaled_action_batch)
        dynamics_loss = F.cross_entropy(
            z_next_pred_logits.reshape(-1, self.config.num_classes),
            z_next_target.reshape(-1, self.config.num_classes).argmax(dim=-1)
        )
        
        # Reward loss (two-hot symlog)
        reward_logits = self.world_model.predict_reward(z_next_pred)
        reward_symlog = symlog(reward_batch)
        reward_target = two_hot_encode(reward_symlog)
        reward_loss = F.cross_entropy(
            reward_logits.reshape(-1, 255),
            reward_target.reshape(-1, 255).argmax(dim=-1)
        )
        
        # Continuation loss
        cont_logits = self.world_model.predict_continuation(z_next_pred)
        cont_target = 1.0 - done_batch
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)
        
        # Teammate prediction loss
        # ★ FIX: agent_name(str) → agent_idx(int) 変換
        teammate_logits_dict = self.world_model.predict_teammates(z, self.agent_idx)
        teammate_loss = 0.0
        count = 0
        
        for batch_idx in range(len(other_actions_batch)):
            for t in range(self.config.wm_batch_length):
                if other_actions_batch[batch_idx][t] is not None:
                    # agent_name → agent_idx に変換
                    other_actions_by_idx = {}
                    for aname, aact in other_actions_batch[batch_idx][t].items():
                        aidx = MATWMAgent.AGENT_NAME_TO_IDX.get(aname)
                        if aidx is not None:
                            other_actions_by_idx[aidx] = aact
                    
                    for other_agent_idx, logits in teammate_logits_dict.items():
                        other_action = other_actions_by_idx.get(other_agent_idx)
                        if other_action is not None:
                            target = torch.LongTensor([other_action]).to(self.device)
                            teammate_loss += F.cross_entropy(
                                logits[batch_idx, t, :].unsqueeze(0), target
                            )
                            count += 1
        
        if count > 0:
            teammate_loss = teammate_loss / count
        else:
            teammate_loss = torch.tensor(0.0).to(self.device)
        
        # KL divergence with free nats
        kl_loss = F.kl_div(
            F.log_softmax(z_logits.reshape(-1, self.config.num_classes), dim=-1),
            F.softmax(z_next_logits_target.reshape(-1, self.config.num_classes), dim=-1),
            reduction='batchmean'
        )
        kl_loss = torch.maximum(kl_loss, torch.tensor(self.config.free_nats).to(self.device))
        
        # Total loss
        total_loss = (
            self.config.recon_weight * recon_loss +
            self.config.dynamics_weight * dynamics_loss +
            self.config.reward_weight * reward_loss +
            self.config.continuation_weight * cont_loss +
            self.config.teammate_weight * teammate_loss +
            self.config.kl_weight * kl_loss
        )
        
        # Update
        self.wm_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.gradient_clip_wm)
        self.wm_optimizer.step()
        
        return {
            'wm_total_loss': total_loss.item(),
            'wm_recon_loss': recon_loss.item(),
            'wm_dynamics_loss': dynamics_loss.item(),
            'wm_reward_loss': reward_loss.item(),
            'wm_cont_loss': cont_loss.item(),
            'wm_teammate_loss': teammate_loss.item() if isinstance(teammate_loss, torch.Tensor) else teammate_loss,
            'wm_kl_loss': kl_loss.item(),
        }
    
    def train_agent(self):
        """
        Train actor-critic with imagination.
        
        ★ FIX: 
          - Critic loss: MSE → Huber loss（スパイク防止）
          - returns をクリップ（発散防止）
          - advantages を正規化（学習安定化）
          - Actor loss にエントロピーボーナス追加（探索促進）
        """
        sequences = self.replay_buffer.sample_random(
            self.config.agent_batch_size, 1  # Sample single starting states
        )
        if sequences is None:
            return {}
        
        # Get starting observations
        obs_start = np.stack([seq[0]['obs'] for seq in sequences])
        obs_start = torch.FloatTensor(obs_start).to(self.device)
        
        # Encode starting state
        with torch.no_grad():
            z, _ = self.world_model.encode(obs_start)
        
        # Imagination rollout
        z_trajectory = [z]
        reward_trajectory = []
        value_trajectory = []
        action_log_prob_trajectory = []
        entropy_trajectory = []          # ★ エントロピー追加
        continuation_trajectory = []
        
        z_current = z
        for t in range(self.config.imagination_horizon):
            # Select action with actor
            action_logits = self.actor(z_current)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
            entropy = action_dist.entropy()  # ★ エントロピー
            
            # Scale action
            scaled_action = action + self.agent_idx * self.config.action_dim
            
            # Predict next state
            z_next, _ = self.world_model.predict_next(
                z_current.unsqueeze(1), scaled_action.unsqueeze(1)
            )
            z_next = z_next.squeeze(1)
            
            # Predict reward and continuation
            reward_logits = self.world_model.predict_reward(z_next)
            reward_dist = F.softmax(reward_logits, dim=-1)
            reward = symexp(two_hot_decode(reward_dist))
            
            cont_logit = self.world_model.predict_continuation(z_next)
            continuation = torch.sigmoid(cont_logit)
            
            # Get value
            value = self.critic(z_current)
            
            # Store
            z_trajectory.append(z_next)
            reward_trajectory.append(reward)
            value_trajectory.append(value)
            action_log_prob_trajectory.append(action_log_prob)
            entropy_trajectory.append(entropy)  # ★
            continuation_trajectory.append(continuation)
            
            z_current = z_next.detach()
        
        # Final value
        final_value = self.critic(z_current)
        
        # Compute GAE
        rewards = torch.stack(reward_trajectory, dim=1)
        values = torch.stack(value_trajectory + [final_value], dim=1)
        continuations = torch.stack(continuation_trajectory, dim=1)
        
        # TD error
        deltas = rewards + self.config.gamma * continuations * values[:, 1:] - values[:, :-1]
        
        # GAE
        advantages = []
        gae = 0
        for t in reversed(range(self.config.imagination_horizon)):
            gae = deltas[:, t] + self.config.gamma * self.config.lambda_gae * continuations[:, t] * gae
            advantages.insert(0, gae)
        advantages = torch.stack(advantages, dim=1)
        
        # ★ FIX: advantages を正規化（学習安定化）
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages_norm = (advantages - adv_mean) / adv_std
        
        # Returns
        returns = advantages + torch.stack(value_trajectory, dim=1)
        # ★ FIX: returns をクリップ（Critic の発散防止）
        returns = torch.clamp(returns, -100.0, 100.0)
        
        # Actor loss (policy gradient + entropy bonus)
        action_log_probs = torch.stack(action_log_prob_trajectory, dim=1)
        entropies = torch.stack(entropy_trajectory, dim=1)
        entropy_coef = 0.01  # ★ エントロピーボーナス係数
        actor_loss = -(action_log_probs * advantages_norm.detach()).mean() - entropy_coef * entropies.mean()
        
        # ★ FIX: Critic loss を Huber loss に変更（スパイク防止）
        critic_loss = F.smooth_l1_loss(
            torch.stack(value_trajectory, dim=1), returns.detach()
        )
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip_agent)
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip_agent)
        self.critic_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_imagined_reward': rewards.mean().item(),
            'mean_value': values.mean().item(),
            'mean_entropy': entropies.mean().item(),       # ★ 新メトリクス
            'mean_advantage': advantages.mean().item(),     # ★ 新メトリクス
        }
    
    def save(self, path):
        """
        Save agent (Actor/Critic only for shared world model setup).
        
        Note: World Model and its optimizer should be saved separately
        when using shared world model architecture.
        """
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }
        
        # Only save world model if this agent owns it (not shared)
        if self.owns_world_model:
            checkpoint['world_model'] = self.world_model.state_dict()
            checkpoint['wm_optimizer'] = self.wm_optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    def load(self, path):
        """
        Load agent (Actor/Critic only for shared world model setup).
        
        Note: World Model should be loaded separately when using
        shared world model architecture.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        # Only load world model if this agent owns it (not shared)
        if self.owns_world_model and 'world_model' in checkpoint:
            self.world_model.load_state_dict(checkpoint['world_model'])
            self.wm_optimizer.load_state_dict(checkpoint['wm_optimizer'])


print("MATWMAgent class loaded successfully!")


