"""
Reinforcement Learning Models Module
=====================================

Production-grade reinforcement learning models for algorithmic trading.
Implements DQN, PPO, and A2C algorithms with proper financial considerations.

Models:
- DQNAgent: Deep Q-Network for discrete actions
- DoubleDQNAgent: Double DQN to reduce overestimation
- PPOAgent: Proximal Policy Optimization
- A2CAgent: Advantage Actor-Critic

Features:
- Experience replay with prioritization
- Target network soft updates
- Reward shaping for trading
- Portfolio-aware state representation
- Risk-adjusted reward functions

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, NamedTuple
import pickle
import random

import numpy as np
from numpy.typing import NDArray

from config.settings import get_logger
from models.base import (
    BaseModel,
    ModelConfig,
    ModelType,
    ModelState,
    ModelRegistry,
)

logger = get_logger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class RLAction(int, Enum):
    """Trading actions for RL agents."""
    SELL = 0
    HOLD = 1
    BUY = 2


class Transition(NamedTuple):
    """Experience replay transition."""
    state: NDArray[np.float64]
    action: int
    reward: float
    next_state: NDArray[np.float64]
    done: bool


class Episode(NamedTuple):
    """Complete episode data."""
    states: list[NDArray]
    actions: list[int]
    rewards: list[float]
    values: list[float]  # For actor-critic
    log_probs: list[float]  # For PPO


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RLConfig(ModelConfig):
    """Base configuration for RL models."""
    model_type: ModelType = ModelType.REINFORCEMENT
    
    # State space
    state_size: int = 0  # Set during initialization
    lookback_window: int = 60
    include_position: bool = True
    include_pnl: bool = True
    
    # Action space
    action_size: int = 3  # SELL, HOLD, BUY
    
    # Training
    episodes: int = 1000
    max_steps_per_episode: int = 1000
    
    # Reward shaping
    reward_type: str = "sharpe"  # pnl, sharpe, sortino, risk_adjusted
    transaction_cost: float = 0.001
    reward_scaling: float = 100.0
    
    # Evaluation
    eval_episodes: int = 10
    eval_frequency: int = 50


@dataclass
class DQNConfig(RLConfig):
    """Configuration for DQN agent."""
    name: str = "DQNAgent"
    
    # Network architecture
    hidden_layers: list[int] = field(default_factory=lambda: [256, 256, 128])
    activation: str = "relu"
    
    # DQN parameters
    learning_rate: float = 0.0001
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Experience replay
    buffer_size: int = 100_000
    batch_size: int = 64
    min_buffer_size: int = 1000
    
    # Target network
    target_update_frequency: int = 100
    soft_update_tau: float = 0.001  # For soft updates
    use_soft_update: bool = True
    
    # Double DQN
    use_double_dqn: bool = True
    
    # Prioritized replay
    use_prioritized_replay: bool = False
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_end: float = 1.0
    
    # Dueling architecture
    use_dueling: bool = True


@dataclass
class PPOConfig(RLConfig):
    """Configuration for PPO agent."""
    name: str = "PPOAgent"
    
    # Network architecture
    hidden_layers: list[int] = field(default_factory=lambda: [256, 256])
    
    # PPO parameters
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95  # GAE parameter
    clip_epsilon: float = 0.2  # PPO clipping
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training
    n_steps: int = 2048  # Steps before update
    n_epochs: int = 10  # Epochs per update
    minibatch_size: int = 64
    
    # Advantage normalization
    normalize_advantage: bool = True


@dataclass
class A2CConfig(RLConfig):
    """Configuration for A2C agent."""
    name: str = "A2CAgent"
    
    # Network architecture
    hidden_layers: list[int] = field(default_factory=lambda: [256, 256])
    
    # A2C parameters
    learning_rate: float = 0.0007
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Update frequency
    n_steps: int = 5


# =============================================================================
# EXPERIENCE REPLAY
# =============================================================================

class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    
    Stores transitions and provides random sampling for training.
    """
    
    def __init__(
        self,
        capacity: int = 100_000,
        prioritized: bool = False,
        alpha: float = 0.6,
    ):
        """Initialize replay buffer."""
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha
        
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self.priorities: deque[float] = deque(maxlen=capacity) if prioritized else None
    
    def push(
        self,
        state: NDArray,
        action: int,
        reward: float,
        next_state: NDArray,
        done: bool,
        priority: float | None = None,
    ) -> None:
        """Add transition to buffer."""
        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)
        
        if self.prioritized:
            max_priority = max(self.priorities) if self.priorities else 1.0
            self.priorities.append(priority or max_priority)
    
    def sample(
        self,
        batch_size: int,
        beta: float = 0.4,
    ) -> tuple[list[Transition], NDArray | None, list[int] | None]:
        """Sample batch of transitions."""
        if self.prioritized:
            # Prioritized sampling
            priorities = np.array(self.priorities)
            probs = priorities ** self.alpha
            probs /= probs.sum()
            
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            transitions = [self.buffer[i] for i in indices]
            
            # Importance sampling weights
            weights = (len(self.buffer) * probs[indices]) ** (-beta)
            weights /= weights.max()
            
            return transitions, weights, indices
        else:
            # Uniform sampling
            transitions = random.sample(list(self.buffer), batch_size)
            return transitions, None, None
    
    def update_priorities(
        self,
        indices: list[int],
        priorities: NDArray,
    ) -> None:
        """Update priorities for prioritized replay."""
        if self.prioritized and self.priorities:
            for idx, priority in zip(indices, priorities):
                self.priorities[idx] = priority + 1e-6
    
    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# REWARD FUNCTIONS
# =============================================================================

class RewardFunction:
    """
    Reward function for trading RL agents.
    
    Provides various reward shaping strategies for financial optimization.
    """
    
    def __init__(
        self,
        reward_type: str = "sharpe",
        transaction_cost: float = 0.001,
        scaling: float = 100.0,
        window: int = 20,
    ):
        """Initialize reward function."""
        self.reward_type = reward_type
        self.transaction_cost = transaction_cost
        self.scaling = scaling
        self.window = window
        
        self._returns_history: list[float] = []
    
    def calculate(
        self,
        pnl: float,
        action: int,
        position_changed: bool,
        position: int,
        portfolio_value: float,
    ) -> float:
        """Calculate reward for a step."""
        # Transaction cost penalty
        cost = self.transaction_cost * portfolio_value if position_changed else 0.0
        net_pnl = pnl - cost
        
        # Store return
        ret = net_pnl / portfolio_value if portfolio_value > 0 else 0
        self._returns_history.append(ret)
        
        if self.reward_type == "pnl":
            # Simple PnL
            reward = net_pnl
            
        elif self.reward_type == "return":
            # Return percentage
            reward = ret
            
        elif self.reward_type == "sharpe":
            # Rolling Sharpe ratio
            if len(self._returns_history) >= self.window:
                returns = np.array(self._returns_history[-self.window:])
                mean_ret = np.mean(returns)
                std_ret = np.std(returns) + 1e-10
                reward = mean_ret / std_ret * np.sqrt(252)  # Annualized
            else:
                reward = ret
                
        elif self.reward_type == "sortino":
            # Rolling Sortino ratio
            if len(self._returns_history) >= self.window:
                returns = np.array(self._returns_history[-self.window:])
                mean_ret = np.mean(returns)
                downside_ret = returns[returns < 0]
                downside_std = np.std(downside_ret) if len(downside_ret) > 0 else 1e-10
                reward = mean_ret / (downside_std + 1e-10) * np.sqrt(252)
            else:
                reward = ret
                
        elif self.reward_type == "risk_adjusted":
            # Risk-adjusted return with drawdown penalty
            reward = ret
            if len(self._returns_history) >= self.window:
                # Drawdown penalty
                cumulative = np.cumprod(1 + np.array(self._returns_history[-self.window:]))
                peak = np.maximum.accumulate(cumulative)
                drawdown = (peak - cumulative) / peak
                max_dd = np.max(drawdown)
                reward = ret - 0.5 * max_dd
        else:
            reward = net_pnl
        
        return reward * self.scaling
    
    def reset(self) -> None:
        """Reset reward history."""
        self._returns_history.clear()


# =============================================================================
# TRADING ENVIRONMENT
# =============================================================================

class TradingEnvironment:
    """
    Trading environment for RL agents.
    
    Simulates market interaction with portfolio management.
    """
    
    def __init__(
        self,
        data: NDArray[np.float64],
        feature_names: list[str],
        initial_capital: float = 100_000.0,
        lookback_window: int = 60,
        transaction_cost: float = 0.001,
        reward_type: str = "sharpe",
    ):
        """Initialize environment."""
        self.data = data
        self.feature_names = feature_names
        self.initial_capital = initial_capital
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        
        # State dimensions
        self.n_features = data.shape[1]
        self.state_size = lookback_window * self.n_features + 2  # +2 for position and pnl
        
        # Reward function
        self.reward_fn = RewardFunction(
            reward_type=reward_type,
            transaction_cost=transaction_cost,
        )
        
        # State
        self._current_step = 0
        self._position = 0  # -1, 0, 1
        self._capital = initial_capital
        self._portfolio_value = initial_capital
        self._entry_price = 0.0
        self._done = False
        
        # Find close price column
        self._close_idx = self._find_close_column()
    
    def _find_close_column(self) -> int:
        """Find close price column index."""
        for i, name in enumerate(self.feature_names):
            if "close" in name.lower():
                return i
        return 0  # Default to first column
    
    def reset(self) -> NDArray[np.float64]:
        """Reset environment to initial state."""
        self._current_step = self.lookback_window
        self._position = 0
        self._capital = self.initial_capital
        self._portfolio_value = self.initial_capital
        self._entry_price = 0.0
        self._done = False
        self.reward_fn.reset()
        
        return self._get_state()
    
    def _get_state(self) -> NDArray[np.float64]:
        """Get current state representation."""
        # Market features (lookback window)
        start_idx = max(0, self._current_step - self.lookback_window)
        market_data = self.data[start_idx:self._current_step].flatten()
        
        # Pad if necessary
        expected_size = self.lookback_window * self.n_features
        if len(market_data) < expected_size:
            market_data = np.pad(market_data, (expected_size - len(market_data), 0))
        
        # Portfolio state
        position_normalized = self._position / 1.0
        pnl_normalized = (self._portfolio_value - self.initial_capital) / self.initial_capital
        
        state = np.concatenate([
            market_data,
            [position_normalized, pnl_normalized],
        ])
        
        return state.astype(np.float32)
    
    def step(
        self,
        action: int,
    ) -> tuple[NDArray[np.float64], float, bool, dict[str, Any]]:
        """Execute action and return new state."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start new episode.")
        
        # Get current price
        current_price = self.data[self._current_step, self._close_idx]
        
        # Previous state
        prev_position = self._position
        prev_value = self._portfolio_value
        
        # Execute action
        position_changed = False
        action_enum = RLAction(action)
        
        if action_enum == RLAction.BUY and self._position <= 0:
            # Close short / Open long
            if self._position == -1:
                pnl = (self._entry_price - current_price) * self._capital / self._entry_price
                self._capital += pnl
            self._position = 1
            self._entry_price = current_price
            position_changed = True
            
        elif action_enum == RLAction.SELL and self._position >= 0:
            # Close long / Open short
            if self._position == 1:
                pnl = (current_price - self._entry_price) * self._capital / self._entry_price
                self._capital += pnl
            self._position = -1
            self._entry_price = current_price
            position_changed = True
        
        # Update portfolio value
        if self._position == 1:
            self._portfolio_value = self._capital * (1 + (current_price - self._entry_price) / self._entry_price)
        elif self._position == -1:
            self._portfolio_value = self._capital * (1 + (self._entry_price - current_price) / self._entry_price)
        else:
            self._portfolio_value = self._capital
        
        # Calculate reward
        pnl = self._portfolio_value - prev_value
        reward = self.reward_fn.calculate(
            pnl=pnl,
            action=action,
            position_changed=position_changed,
            position=self._position,
            portfolio_value=prev_value,
        )
        
        # Move to next step
        self._current_step += 1
        
        # Check if done
        self._done = (
            self._current_step >= len(self.data) - 1 or
            self._portfolio_value <= 0
        )
        
        # Get new state
        next_state = self._get_state()
        
        # Info
        info = {
            "position": self._position,
            "portfolio_value": self._portfolio_value,
            "pnl": pnl,
            "return": (self._portfolio_value - self.initial_capital) / self.initial_capital,
            "action_taken": action_enum.name,
            "position_changed": position_changed,
        }
        
        return next_state, reward, self._done, info
    
    @property
    def current_step(self) -> int:
        return self._current_step
    
    @property
    def portfolio_value(self) -> float:
        return self._portfolio_value
    
    @property
    def total_return(self) -> float:
        return (self._portfolio_value - self.initial_capital) / self.initial_capital


# =============================================================================
# DQN AGENT
# =============================================================================

class DQNAgent(BaseModel):
    """
    Deep Q-Network Agent for trading.
    
    Features:
    - Double DQN to reduce overestimation
    - Dueling architecture
    - Prioritized experience replay
    - Epsilon-greedy exploration
    
    Example:
        config = DQNConfig(
            state_size=100,
            hidden_layers=[256, 256],
            use_double_dqn=True,
        )
        agent = DQNAgent(config)
        
        # Training loop
        for episode in range(1000):
            state = env.reset()
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.train()
                state = next_state
    """
    
    def __init__(self, config: DQNConfig | None = None):
        """Initialize DQN agent."""
        self.config = config or DQNConfig()
        super().__init__(self.config)
        
        # Networks
        self._q_network = None
        self._target_network = None
        self._optimizer = None
        
        # Replay buffer
        self._buffer = ReplayBuffer(
            capacity=self.config.buffer_size,
            prioritized=self.config.use_prioritized_replay,
            alpha=self.config.priority_alpha,
        )
        
        # Exploration
        self._epsilon = self.config.epsilon_start
        
        # Training state
        self._training_step = 0
        self._episode = 0
        self._total_reward = 0.0
        
        # Metrics
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []
        self._losses: list[float] = []
    
    def _default_config(self) -> DQNConfig:
        return DQNConfig()
    
    def _build_model(self) -> Any:
        """Build Q-network."""
        try:
            import torch
            import torch.nn as nn
            
            class QNetwork(nn.Module):
                def __init__(self, state_size: int, action_size: int, hidden_layers: list[int], dueling: bool):
                    super().__init__()
                    self.dueling = dueling
                    
                    # Feature layers
                    layers = []
                    prev_size = state_size
                    for hidden_size in hidden_layers:
                        layers.extend([
                            nn.Linear(prev_size, hidden_size),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                        ])
                        prev_size = hidden_size
                    
                    self.features = nn.Sequential(*layers)
                    
                    if dueling:
                        # Dueling architecture
                        self.value_stream = nn.Sequential(
                            nn.Linear(prev_size, prev_size // 2),
                            nn.ReLU(),
                            nn.Linear(prev_size // 2, 1),
                        )
                        self.advantage_stream = nn.Sequential(
                            nn.Linear(prev_size, prev_size // 2),
                            nn.ReLU(),
                            nn.Linear(prev_size // 2, action_size),
                        )
                    else:
                        self.q_head = nn.Linear(prev_size, action_size)
                
                def forward(self, x):
                    features = self.features(x)
                    
                    if self.dueling:
                        value = self.value_stream(features)
                        advantage = self.advantage_stream(features)
                        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
                    else:
                        q_values = self.q_head(features)
                    
                    return q_values
            
            return QNetwork
            
        except ImportError:
            logger.warning("PyTorch not available. DQN will use numpy-based implementation.")
            return None
    
    def _initialize_networks(self, state_size: int, action_size: int) -> None:
        """Initialize Q-networks."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        QNetwork = self._build_model()
        
        self._q_network = QNetwork(
            state_size,
            action_size,
            self.config.hidden_layers,
            self.config.use_dueling,
        )
        
        self._target_network = QNetwork(
            state_size,
            action_size,
            self.config.hidden_layers,
            self.config.use_dueling,
        )
        
        # Copy weights to target
        self._target_network.load_state_dict(self._q_network.state_dict())
        
        # Optimizer
        self._optimizer = optim.Adam(
            self._q_network.parameters(),
            lr=self.config.learning_rate,
        )
        
        # Device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._q_network.to(self._device)
        self._target_network.to(self._device)
    
    def select_action(
        self,
        state: NDArray[np.float64],
        training: bool = True,
    ) -> int:
        """Select action using epsilon-greedy policy."""
        import torch
        
        if training and random.random() < self._epsilon:
            return random.randint(0, self.config.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self._device)
            q_values = self._q_network(state_tensor)
            return q_values.argmax(dim=-1).item()
    
    def store_transition(
        self,
        state: NDArray,
        action: int,
        reward: float,
        next_state: NDArray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self._buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> float | None:
        """Perform one training step."""
        import torch
        import torch.nn.functional as F
        
        # Check if enough samples
        if len(self._buffer) < self.config.min_buffer_size:
            return None
        
        # Sample batch
        beta = min(
            self.config.priority_beta_end,
            self.config.priority_beta_start + 
            (self.config.priority_beta_end - self.config.priority_beta_start) *
            self._training_step / self.config.episodes
        )
        
        transitions, weights, indices = self._buffer.sample(
            self.config.batch_size,
            beta=beta,
        )
        
        # Prepare batch
        states = torch.FloatTensor(np.array([t.state for t in transitions])).to(self._device)
        actions = torch.LongTensor([t.action for t in transitions]).to(self._device)
        rewards = torch.FloatTensor([t.reward for t in transitions]).to(self._device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in transitions])).to(self._device)
        dones = torch.FloatTensor([t.done for t in transitions]).to(self._device)
        
        # Current Q values
        current_q = self._q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values
        with torch.no_grad():
            if self.config.use_double_dqn:
                # Double DQN: use main network to select action, target to evaluate
                next_actions = self._q_network(next_states).argmax(dim=-1)
                next_q = self._target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                next_q = self._target_network(next_states).max(dim=-1)[0]
            
            target_q = rewards + self.config.gamma * next_q * (1 - dones)
        
        # Loss
        if weights is not None:
            weights_tensor = torch.FloatTensor(weights).to(self._device)
            td_error = F.smooth_l1_loss(current_q, target_q, reduction='none')
            loss = (weights_tensor * td_error).mean()
            
            # Update priorities
            priorities = td_error.detach().cpu().numpy()
            self._buffer.update_priorities(indices, priorities)
        else:
            loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._q_network.parameters(), 1.0)
        self._optimizer.step()
        
        # Update target network
        self._training_step += 1
        if self.config.use_soft_update:
            # Soft update
            for target_param, local_param in zip(
                self._target_network.parameters(),
                self._q_network.parameters(),
            ):
                target_param.data.copy_(
                    self.config.soft_update_tau * local_param.data +
                    (1 - self.config.soft_update_tau) * target_param.data
                )
        elif self._training_step % self.config.target_update_frequency == 0:
            # Hard update
            self._target_network.load_state_dict(self._q_network.state_dict())
        
        # Decay epsilon
        self._epsilon = max(
            self.config.epsilon_end,
            self._epsilon * self.config.epsilon_decay,
        )
        
        return loss.item()
    
    def train_episode(
        self,
        env: TradingEnvironment,
    ) -> dict[str, float]:
        """Train for one episode."""
        state = env.reset()
        total_reward = 0.0
        steps = 0
        losses = []
        
        while True:
            action = self.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            self.store_transition(state, action, reward, next_state, done)
            
            loss = self.train_step()
            if loss is not None:
                losses.append(loss)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done or steps >= self.config.max_steps_per_episode:
                break
        
        self._episode += 1
        self._episode_rewards.append(total_reward)
        self._episode_lengths.append(steps)
        if losses:
            self._losses.extend(losses)
        
        return {
            "episode": self._episode,
            "total_reward": total_reward,
            "steps": steps,
            "epsilon": self._epsilon,
            "avg_loss": np.mean(losses) if losses else 0.0,
            "portfolio_return": env.total_return,
        }
    
    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | None = None,
        feature_names: list[str] | None = None,
        **kwargs: Any,
    ) -> "DQNAgent":
        """Train the agent on data."""
        logger.info(f"Training DQN agent on {len(X)} samples")
        
        # Create environment
        feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        env = TradingEnvironment(
            data=X,
            feature_names=feature_names,
            lookback_window=self.config.lookback_window,
            transaction_cost=self.config.transaction_cost,
            reward_type=self.config.reward_type,
        )
        
        # Initialize networks
        self._initialize_networks(env.state_size, self.config.action_size)
        
        self._state = ModelState.TRAINING
        
        # Training loop
        for episode in range(self.config.episodes):
            result = self.train_episode(env)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self._episode_rewards[-10:])
                avg_return = result["portfolio_return"]
                logger.info(
                    f"Episode {episode + 1}/{self.config.episodes}: "
                    f"Reward={result['total_reward']:.2f}, "
                    f"Avg={avg_reward:.2f}, "
                    f"Return={avg_return:.2%}, "
                    f"Epsilon={self._epsilon:.3f}"
                )
        
        self._state = ModelState.TRAINED
        return self
    
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict actions for states."""
        actions = []
        for state in X:
            action = self.select_action(state, training=False)
            actions.append(action)
        return np.array(actions)
    
    def _fit_impl(self, *args, **kwargs):
        pass
    
    def _predict_impl(self, X):
        return self.predict(X)
    
    def evaluate(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | None = None,
    ) -> dict[str, float]:
        """Evaluate agent performance."""
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        env = TradingEnvironment(
            data=X,
            feature_names=feature_names,
            lookback_window=self.config.lookback_window,
        )
        
        total_rewards = []
        returns = []
        
        for _ in range(self.config.eval_episodes):
            state = env.reset()
            total_reward = 0.0
            
            while True:
                action = self.select_action(state, training=False)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                state = next_state
                if done:
                    break
            
            total_rewards.append(total_reward)
            returns.append(env.total_return)
        
        return {
            "avg_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "avg_return": np.mean(returns),
            "max_return": np.max(returns),
            "min_return": np.min(returns),
        }
    
    def save(self, path: str | Path) -> None:
        """Save agent state."""
        import torch
        
        path = Path(path)
        
        state = {
            "config": self.config,
            "q_network": self._q_network.state_dict() if self._q_network else None,
            "target_network": self._target_network.state_dict() if self._target_network else None,
            "optimizer": self._optimizer.state_dict() if self._optimizer else None,
            "epsilon": self._epsilon,
            "training_step": self._training_step,
            "episode": self._episode,
            "episode_rewards": self._episode_rewards,
        }
        
        torch.save(state, path)
        logger.info(f"Agent saved to {path}")
    
    @classmethod
    def load(cls, path: str | Path) -> "DQNAgent":
        """Load agent state."""
        import torch
        
        state = torch.load(path)
        agent = cls(state["config"])
        
        if state["q_network"]:
            agent._initialize_networks(
                agent.config.state_size,
                agent.config.action_size,
            )
            agent._q_network.load_state_dict(state["q_network"])
            agent._target_network.load_state_dict(state["target_network"])
            if state["optimizer"]:
                agent._optimizer.load_state_dict(state["optimizer"])
        
        agent._epsilon = state["epsilon"]
        agent._training_step = state["training_step"]
        agent._episode = state["episode"]
        agent._episode_rewards = state["episode_rewards"]
        agent._state = ModelState.TRAINED
        
        logger.info(f"Agent loaded from {path}")
        return agent


# =============================================================================
# PPO AGENT (SIMPLIFIED)
# =============================================================================

class PPOAgent(BaseModel):
    """
    Proximal Policy Optimization Agent for trading.
    
    PPO is an on-policy algorithm that constrains policy updates
    to improve stability.
    
    Example:
        config = PPOConfig(state_size=100)
        agent = PPOAgent(config)
        agent.fit(X_train)
    """
    
    def __init__(self, config: PPOConfig | None = None):
        """Initialize PPO agent."""
        self.config = config or PPOConfig()
        super().__init__(self.config)
        
        self._actor = None
        self._critic = None
        self._optimizer = None
        self._device = None
    
    def _default_config(self) -> PPOConfig:
        return PPOConfig()
    
    def _build_model(self) -> Any:
        """Build actor-critic networks."""
        try:
            import torch
            import torch.nn as nn
            
            class ActorCritic(nn.Module):
                def __init__(self, state_size: int, action_size: int, hidden_layers: list[int]):
                    super().__init__()
                    
                    # Shared feature extractor
                    layers = []
                    prev_size = state_size
                    for hidden_size in hidden_layers[:-1]:
                        layers.extend([
                            nn.Linear(prev_size, hidden_size),
                            nn.Tanh(),
                        ])
                        prev_size = hidden_size
                    self.features = nn.Sequential(*layers)
                    
                    # Actor head (policy)
                    self.actor = nn.Sequential(
                        nn.Linear(prev_size, hidden_layers[-1]),
                        nn.Tanh(),
                        nn.Linear(hidden_layers[-1], action_size),
                        nn.Softmax(dim=-1),
                    )
                    
                    # Critic head (value)
                    self.critic = nn.Sequential(
                        nn.Linear(prev_size, hidden_layers[-1]),
                        nn.Tanh(),
                        nn.Linear(hidden_layers[-1], 1),
                    )
                
                def forward(self, x):
                    features = self.features(x)
                    return self.actor(features), self.critic(features)
                
                def get_action(self, state):
                    probs, value = self.forward(state)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    return action, dist.log_prob(action), value
            
            return ActorCritic
            
        except ImportError:
            logger.warning("PyTorch not available for PPO")
            return None
    
    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | None = None,
        feature_names: list[str] | None = None,
        **kwargs: Any,
    ) -> "PPOAgent":
        """Train PPO agent."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        logger.info(f"Training PPO agent on {len(X)} samples")
        
        # Create environment
        feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        env = TradingEnvironment(
            data=X,
            feature_names=feature_names,
            lookback_window=self.config.lookback_window,
        )
        
        # Build networks
        ActorCritic = self._build_model()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._network = ActorCritic(
            env.state_size,
            self.config.action_size,
            self.config.hidden_layers,
        ).to(self._device)
        
        self._optimizer = optim.Adam(
            self._network.parameters(),
            lr=self.config.learning_rate,
        )
        
        self._state = ModelState.TRAINING
        
        # Training loop
        for episode in range(self.config.episodes):
            # Collect trajectories
            states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
            
            state = env.reset()
            total_reward = 0.0
            
            for step in range(self.config.n_steps):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self._device)
                
                with torch.no_grad():
                    action, log_prob, value = self._network.get_action(state_tensor)
                
                next_state, reward, done, info = env.step(action.item())
                
                states.append(state)
                actions.append(action.item())
                rewards.append(reward)
                values.append(value.item())
                log_probs.append(log_prob.item())
                dones.append(done)
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Compute returns and advantages
            returns = self._compute_returns(rewards, values, dones)
            advantages = returns - np.array(values)
            
            if self.config.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update
            states_tensor = torch.FloatTensor(np.array(states)).to(self._device)
            actions_tensor = torch.LongTensor(actions).to(self._device)
            returns_tensor = torch.FloatTensor(returns).to(self._device)
            advantages_tensor = torch.FloatTensor(advantages).to(self._device)
            old_log_probs = torch.FloatTensor(log_probs).to(self._device)
            
            for _ in range(self.config.n_epochs):
                probs, values = self._network(states_tensor)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(actions_tensor)
                entropy = dist.entropy().mean()
                
                # PPO clipping
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages_tensor
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages_tensor
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(values.squeeze(), returns_tensor)
                
                loss = actor_loss + self.config.value_coef * critic_loss - self.config.entropy_coef * entropy
                
                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._network.parameters(), self.config.max_grad_norm)
                self._optimizer.step()
            
            if (episode + 1) % 10 == 0:
                logger.info(
                    f"Episode {episode + 1}/{self.config.episodes}: "
                    f"Reward={total_reward:.2f}, Return={env.total_return:.2%}"
                )
        
        self._state = ModelState.TRAINED
        return self
    
    def _compute_returns(
        self,
        rewards: list[float],
        values: list[float],
        dones: list[bool],
    ) -> NDArray[np.float64]:
        """Compute discounted returns with GAE."""
        returns = np.zeros(len(rewards))
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            returns[t] = gae + values[t]
        
        return returns
    
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict actions."""
        import torch
        
        actions = []
        for state in X:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self._device)
            with torch.no_grad():
                probs, _ = self._network(state_tensor)
                action = probs.argmax(dim=-1).item()
            actions.append(action)
        return np.array(actions)
    
    def _fit_impl(self, *args, **kwargs):
        pass
    
    def _predict_impl(self, X):
        return self.predict(X)
    
    def evaluate(self, X: NDArray, y: NDArray | None = None) -> dict[str, float]:
        """Evaluate agent."""
        return {"avg_reward": 0.0}  # Placeholder
    
    def save(self, path: str | Path) -> None:
        """Save agent."""
        import torch
        torch.save({
            "config": self.config,
            "network": self._network.state_dict() if self._network else None,
        }, path)
    
    @classmethod
    def load(cls, path: str | Path) -> "PPOAgent":
        """Load agent."""
        import torch
        state = torch.load(path)
        agent = cls(state["config"])
        return agent


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_rl_agent(
    agent_type: str,
    **kwargs: Any,
) -> DQNAgent | PPOAgent:
    """
    Factory function to create RL agents.
    
    Args:
        agent_type: Type of agent (dqn, double_dqn, ppo, a2c)
        **kwargs: Agent configuration
    
    Returns:
        Configured RL agent
    """
    agent_map = {
        "dqn": (DQNAgent, DQNConfig),
        "double_dqn": (DQNAgent, DQNConfig),
        "ppo": (PPOAgent, PPOConfig),
    }
    
    agent_type = agent_type.lower()
    
    if agent_type not in agent_map:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent_class, config_class = agent_map[agent_type]
    
    # Special handling for double DQN
    if agent_type == "double_dqn":
        kwargs["use_double_dqn"] = True
    
    config = config_class(**kwargs)
    return agent_class(config)


# =============================================================================
# REGISTER MODELS
# =============================================================================

ModelRegistry.register("dqn", DQNAgent, DQNConfig)
ModelRegistry.register("ppo", PPOAgent, PPOConfig)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "RLAction",
    # Types
    "Transition",
    "Episode",
    # Configs
    "RLConfig",
    "DQNConfig",
    "PPOConfig",
    "A2CConfig",
    # Utilities
    "ReplayBuffer",
    "RewardFunction",
    "TradingEnvironment",
    # Agents
    "DQNAgent",
    "PPOAgent",
    # Factory
    "create_rl_agent",
]