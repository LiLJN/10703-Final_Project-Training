import os
import json
import argparse
import random

import numpy as np
import gymnasium as gym
from tqdm import tqdm
from PIL import Image
import mujoco
from stable_baselines3 import SAC


def generate_random_goalpoint():
    x = 10
    y = 10
    while x**2 + y**2 > 1.0:
        x = random.uniform(-1, 1)
        y = random.uniform(-0.5, 0.5)
    return np.asarray([x, y])


def balanced_binary_array(n: int) -> np.ndarray:
    # 1/4 episodes random action, 3/4 episodes policy action
    arr = np.array([0] * (n // 4) + [1] * (n // 4 * 3))
    np.random.shuffle(arr)
    print(arr)
    return arr


def collect_pusher_dataset(
    save_dir: str = "pusher_vlm_data",
    num_episodes: int = 2000,
    max_steps: int = 200,
    model_path: str = "sac_pusher.zip",
):
    # ---------- Camera config (matches your cam.lookat / distance / az / el) ----------
    cam_config = {
        "lookat": np.array([0.0, 0.0, 0.0]),
        "distance": 3.0,
        "azimuth": 90.0,
        "elevation": -90.0,
        # optional keys that gym’s mujoco envs understand:
        # "trackbodyid": -1,
    }

    # Gymnasium MuJoCo env with rgb_array rendering + custom camera
    env = gym.make(
        "Pusher-v5",
        render_mode="rgb_array",
        default_camera_config=cam_config,
    )

    policy_model = SAC.load(model_path, env=env)

    # ---------- Output dirs ----------
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    meta_path = os.path.join(save_dir, "meta.jsonl")

    total_samples = 0
    toggle_random_action = balanced_binary_array(num_episodes)

    # Cache model & data once
    model = env.unwrapped.model
    data = env.unwrapped.data

    # Find goal body once (no need to search every episode)
    goal_body_id = None
    for i in range(model.nbody):
        name = model.body(i).name
        if name is not None and "goal" in name.lower():
            goal_body_id = i
            break
    if goal_body_id is None:
        raise RuntimeError("Could not find goal body in model.")

    with open(meta_path, "w") as f:
        for ep in tqdm(range(num_episodes), desc="Collecting Episodes"):
            toggle_random_action_flag = toggle_random_action[ep]

            # Reset env
            obs, _ = env.reset()
            done = False

            # ---------- Randomize goal position ----------
            new_goal_xy = generate_random_goalpoint()
            model.body_pos[goal_body_id, 0] = new_goal_xy[0]
            model.body_pos[goal_body_id, 1] = new_goal_xy[1]

            # Recompute kinematics after modifying body_pos
            mujoco.mj_forward(model, data)

            # ---------- Initial obs + frame ----------
            prev_obs = env.unwrapped._get_obs()
            frame = env.render()  # uses the camera from default_camera_config
            img_pil = Image.fromarray(frame)

            step_count = 0
            img_name = f"ep{ep:04d}_step{step_count:03d}.png"
            img_path = os.path.join(img_dir, img_name)
            img_pil.save(img_path)

            fingertip_obs = np.asarray(prev_obs, dtype=float).tolist()[14:17]
            object_obs = np.asarray(prev_obs, dtype=float).tolist()[17:20]
            goal_obs = np.asarray(prev_obs, dtype=float).tolist()[20:]

            # ---------- Rollout ----------
            for _ in tqdm(range(max_steps - 1), desc=f"Ep {ep}", leave=False):
                if done:
                    break

                if toggle_random_action_flag == 0:
                    # Random action
                    action = env.action_space.sample()
                else:
                    # Policy action
                    action, _ = policy_model.predict(prev_obs, deterministic=True)

                curr_obs, reward, terminated, _, _ = env.step(action)
                step_count += 1
                done = terminated

                curr_fingertip_obs = np.asarray(curr_obs, dtype=float).tolist()[14:17]
                curr_object_obs = np.asarray(curr_obs, dtype=float).tolist()[17:20]
                curr_goal_obs = np.asarray(curr_obs, dtype=float).tolist()[20:]

                # Render with same camera config
                curr_frame = env.render()
                curr_img_pil = Image.fromarray(curr_frame)

                curr_img_name = f"ep{ep:04d}_step{step_count:03d}.png"
                curr_img_path = os.path.join(img_dir, curr_img_name)
                curr_img_pil.save(curr_img_path)

                record = {
                    "prev_image": img_name,
                    "curr_image": curr_img_name,
                    "episode": ep,
                    "step": step_count,
                    "reward": float(reward),
                    "prev_fingertip_obs": fingertip_obs,
                    "prev_object_obs": object_obs,
                    "prev_goal_obs": goal_obs,
                    "curr_fingertip_obs": curr_fingertip_obs,
                    "curr_object_obs": curr_object_obs,
                    "curr_goal_obs": curr_goal_obs,
                    "action": np.asarray(action, dtype=float).tolist(),
                    "goal_xy": new_goal_xy.tolist(),
                }

                f.write(json.dumps(record) + "\n")
                total_samples += 1

                # Roll forward state for next step
                prev_obs = curr_obs
                fingertip_obs = curr_fingertip_obs
                object_obs = curr_object_obs
                goal_obs = curr_goal_obs
                frame = curr_frame
                img_name = curr_img_name
                img_path = curr_img_path
                img_pil = curr_img_pil

    env.close()
    print(f"\nDataset collection complete! Saved {total_samples} samples to {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="pusher_vlm_data")
    parser.add_argument("--num_episodes", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()

    collect_pusher_dataset(
        save_dir=args.save_dir,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
    )