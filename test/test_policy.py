import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3 import TD3
import numpy as np
import random
import mujoco

MODEL_PATH = "sac_pusher_random_goal.zip"
# MODEL_PATH = "td3_pusher_random_goal.zip"
ENV_ID = "Pusher-v5"

def generate_random_goalpoint():
    x = 10
    y = 10
    while x**2 + y**2 > 1.0:
        x = random.uniform(-1, 1)
        y = random.uniform(-0.5, 0.5)
    return np.asarray([x, y])

def run_trained_agent(num_episodes=5):
    # Load trained model
    model = SAC.load(MODEL_PATH)
    # model = TD3.load(MODEL_PATH)

    # Create a *non-vectorized* env for playback
    env = gym.make(ENV_ID, render_mode="human")
    

    for ep in range(num_episodes):
        obs, _ = env.reset()
        env_model = env.unwrapped.model
        new_goal_xy = generate_random_goalpoint()
        goal_body_id = None
        for i in range(env_model.nbody):
            name = env_model.body(i).name
            if name is not None and "goal" in name.lower():
                goal_body_id = i
                break
        if goal_body_id is None:
            raise RuntimeError("Could not find goal body in model.")
        env_model.body_pos[goal_body_id, 0] = new_goal_xy[0]
        env_model.body_pos[goal_body_id, 1] = new_goal_xy[1]
        mujoco.mj_forward(env_model, env.unwrapped.data)
        obs = env.unwrapped._get_obs()
        done = False
        ep_reward = 0.0
        steps = 0

        while steps < 200 and not done:
            # SAC predict: deterministic=True for clean behavior
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            ep_reward += reward

            # Render to screen (MuJoCo viewer)
            env.render()
            steps += 1

        print(f"Episode {ep}: total reward = {ep_reward:.2f}")

    env.close()

if __name__ == "__main__":
    run_trained_agent()