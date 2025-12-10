# train_sac_pusher.py
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise

import mujoco  # <-- IMPORTANT: for mj_forward

ENV_ID = "Pusher-v5"
MODEL_PATH = "sac_pusher_random_goal.zip"


# --------- helper: sample random goal in workspace ---------
def generate_random_goalpoint():
    # Your previous logic: sample in region x^2 + y^2 <= 1, y in [-0.5, 0.5]
    x = 10.0
    y = 10.0
    while x**2 + y**2 > 1.0:
        x = np.random.uniform(-1.0, 1.0)
        y = np.random.uniform(-0.5, 0.5)
    return np.array([x, y], dtype=np.float32)

# --------- wrapper: randomize goal at every reset ---------
class RandomGoalPusherEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._goal_body_id = None

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

        # Set local body position of goal (x, y)
        model.body_pos[goal_body_id, 0] = new_goal_xy[0]
        model.body_pos[goal_body_id, 1] = new_goal_xy[1]

        # Recompute MuJoCo forward kinematics / sensors
        mujoco.mj_forward(model, data)

        # Store for debugging / logging if you want
        self.current_goal_xy = new_goal_xy

    def reset(self, **kwargs):
        # Standard reset
        obs, info = self.env.reset(**kwargs)

        # Randomize goal AFTER reset each episode
        self._randomize_goal()

        # Recompute obs so it matches the new goal
        obs = self.env.unwrapped._get_obs()

        return obs, info


# --------- env factory for SB3 (vectorized) ---------
def make_env():
    def _init():
        base_env = gym.make(ENV_ID)
        wrapped_env = RandomGoalPusherEnv(base_env)
        return wrapped_env
    return _init


def main():
    # Vectorized env for SB3: 4 parallel randomized-goal envs
    vec_env = DummyVecEnv([make_env() for _ in range(4)])

    n_actions = vec_env.action_space.shape[0]
    noise_mean = np.zeros(n_actions)
    noise_std = 0.1 * np.ones(n_actions)
    # action_noise = NormalActionNoise(noise_mean, noise_std)

    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        action_noise=None,
        tensorboard_log="./sac_pusher_tb_random_goal",
    )

    model.learn(total_timesteps=int(5e6), progress_bar=True)

    # For evaluation, use the same randomized-goal wrapper
    eval_env = DummyVecEnv([make_env() for _ in range(4)])
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward (random goals): {mean_reward:.2f}")

    model.save(MODEL_PATH)
    print(f"Saved SAC model to {MODEL_PATH}")


if __name__ == "__main__":
    main()