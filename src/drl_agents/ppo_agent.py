from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from .base_agent import BaseAgent

class PPOAgent(BaseAgent):
    """
    PPO Agent wrapper using stable-baselines3, compatible with BaseAgent interface.
    """
    def __init__(self, env, policy='MlpPolicy', **kwargs):
        self.env = env
        self.model = PPO(policy, env, **kwargs)

    def train(self, total_timesteps=10000, **kwargs):
        self.model.learn(total_timesteps=total_timesteps, **kwargs)

    def predict(self, state, deterministic=True):
        action, _ = self.model.predict(state, deterministic=deterministic)
        return action

    def save(self, filepath: str):
        self.model.save(filepath)

    def load(self, filepath: str):
        from stable_baselines3 import PPO
        self.model = PPO.load(filepath, env=self.env)
