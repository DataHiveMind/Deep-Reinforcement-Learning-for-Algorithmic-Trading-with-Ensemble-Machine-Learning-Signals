import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional

class TradingEnv(gym.Env):
    """
    Custom OpenAI Gymnasium-compatible trading environment.
    Simulates the interaction between the DRL agent and the market.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, 
                 data: np.ndarray, 
                 ml_signals: Optional[np.ndarray] = None,
                 initial_cash: float = 10000.0,
                 transaction_cost: float = 0.001,
                 max_position_size: int = 100,
                 max_steps: int = 1000):
        super().__init__()
        self.data = data
        self.ml_signals = ml_signals
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.max_steps = max_steps
        self.current_step = 0
        self.current_cash = initial_cash
        self.current_position = 0
        self.done = False
        # Observation: price features + ML signals (if any)
        obs_dim = data.shape[1]
        if ml_signals is not None:
            obs_dim += ml_signals.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        self.history = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_cash = self.initial_cash
        self.current_position = 0
        self.done = False
        self.history = []
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        if self.done:
            raise RuntimeError("Cannot step in a finished episode.")
        reward = 0.0
        price = self.data[self.current_step][0] if self.current_step < len(self.data) else 0.0
        if action == 0:  # Buy
            if self.current_cash > 0:
                position_size = min(self.max_position_size, int(self.current_cash / (price + self.transaction_cost)))
                cost = position_size * (price + self.transaction_cost)
                self.current_position += position_size
                self.current_cash -= cost
                reward = -self.transaction_cost * position_size
        elif action == 1:  # Sell
            if self.current_position > 0:
                position_size = min(self.max_position_size, self.current_position)
                proceeds = position_size * (price - self.transaction_cost)
                self.current_position -= position_size
                self.current_cash += proceeds
                reward = -self.transaction_cost * position_size
        elif action == 2:  # Hold
            reward = 0.0
        else:
            raise ValueError("Invalid action. Action must be 0 (Buy), 1 (Sell), or 2 (Hold).")
        self.current_step += 1
        self.done = self.current_step >= self.max_steps or self.current_step >= len(self.data)
        obs = self._get_obs()
        info = self._get_info()
        portfolio_value = self.current_cash + self.current_position * price
        self.history.append({
            'step': self.current_step,
            'cash': self.current_cash,
            'position': self.current_position,
            'reward': reward,
            'portfolio_value': portfolio_value
        })
        terminated = self.done
        truncated = self.current_step >= self.max_steps and not self.done
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Cash: {self.current_cash:.2f}, Position: {self.current_position}, Done: {self.done}")
        print(self.history[-5:] if len(self.history) > 0 else "No history yet.")

    def _get_obs(self):
        obs = self.data[self.current_step] if self.current_step < len(self.data) else np.zeros(self.data.shape[1])
        if self.ml_signals is not None:
            ml_obs = self.ml_signals[self.current_step] if self.current_step < len(self.ml_signals) else np.zeros(self.ml_signals.shape[1])
            obs = np.concatenate([obs, ml_obs])
        return obs.astype(np.float32)

    def _get_info(self):
        price = self.data[self.current_step][0] if self.current_step < len(self.data) and len(self.data[self.current_step]) > 0 else 0.0
        portfolio_value = self.current_cash + self.current_position * price
        return {
            "step": self.current_step,
            "cash": self.current_cash,
            "position": self.current_position,
            "portfolio_value": portfolio_value,
        }
