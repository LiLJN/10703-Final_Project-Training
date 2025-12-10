import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy

import mujoco

ENV_ID = "Pusher-v5"
MODEL_PATH = "ppo_pusher_random_goal.zip"

def generate_random_goalpoint():
    x = 10.0
    y = 10.0
    while x**2 + y**2 > 1.0:
        x = np.random.uniform(-1.0, 1.0)
        y = np.random.uniform(-0.5, 0.5)
    return np.array([x, y], dtype=np.float32)

class RandomGoalPusherEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._goal_body_id = None
        self.current_goal_xy = None

    def _find_goal_body(self):
        if self._goal_body_id is not None:
            return self._goal_body_id
        model = self.env.unwrapped.model
        for i in range(model.nbody):
            name = model.body(i).name
            if name is not None and "goal" in name.lower():
                self._goal_body_id = i
                break
        if self._goal_body_id is None:
            raise RuntimeError("Could not find goal body in model.")
        return self._goal_body_id

    def _randomize_goal(self):
        model = self.env.unwrapped.model
        data = self.env.unwrapped.data
        goal_body_id = self._find_goal_body()
        new_goal_xy = generate_random_goalpoint()
        model.body_pos[goal_body_id, 0] = new_goal_xy[0]
        model.body_pos[goal_body_id, 1] = new_goal_xy[1]
        mujoco.mj_forward(model, data)
        self.current_goal_xy = new_goal_xy

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._randomize_goal()
        obs = self.env.unwrapped._get_obs()
        return obs, info

def make_env():
    def _init():
        base_env = gym.make(ENV_ID)
        return RandomGoalPusherEnv(base_env)
    return _init

def main():
    vec_env = DummyVecEnv([make_env() for _ in range(8)]) 
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="./ppo_pusher_tb",
        n_steps=2048, 
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        learning_rate=3e-4,
        clip_range=0.2,
    )

    model.learn(total_timesteps=int(5e6), progress_bar=True)

    eval_env = DummyVecEnv([make_env() for _ in range(2)])
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"[PPO] Mean reward (random goals): {mean_reward:.2f}")

    model.save(MODEL_PATH)
    print(f"Saved TD3 model to {MODEL_PATH}")

if __name__ == "__main__":
    main()