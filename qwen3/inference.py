#!/usr/bin/env python3
import re
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image
import torch

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# -------------------------
# Paths & basic config
# -------------------------
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent

LORA_DIR = ROOT_DIR / "qwen3_vl_pusher_lora" 

BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

ENV_ID = "Pusher-v5"
NUM_EPISODES = 2
MAX_STEPS_PER_EP = 200
MAX_SAMPLES = 200  # max (true, pred) pairs; set None for unlimited


# Messages copied from your *new* dataset format
SYSTEM_TEXT = (
    "You are a VLM that takes 2 consecutive frames from the 'Pusher-V5' "
    "environment and generate rewards base on them."
)

USER_TEXT = (
    "The first image shows the scene at the previous time step; the second image "
    "shows the scene at the current time step. From these two images, estimate the "
    "immediate reward at the current time step. Please output a single float number."
)


# -------------------------
# Utils
# -------------------------
def get_device() -> str:
    # if torch.backends.mps.is_available():
    #     return "mps"
    return "cpu"


def pil_from_array(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def parse_reward(raw: str):
    raw = raw.strip()
    # try direct float
    try:
        return float(raw)
    except ValueError:
        pass
    # fallback: first float substring
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            pass
    return None

# -------------------------
# Model loading
# -------------------------
def load_model_and_processor():
    device = get_device()
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Base model: {BASE_MODEL_ID}")
    print(f"[INFO] LoRA dir:   {LORA_DIR}")

    if not LORA_DIR.is_dir():
        raise FileNotFoundError(f"LoRA directory not found: {LORA_DIR}")

    print("[INFO] Loading base Qwen3-VL model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,   # CPU / MPS friendly
        trust_remote_code=True,
    )

    print("[INFO] Loading LoRA adapter into base model...")
    model = PeftModel.from_pretrained(
        model,
        LORA_DIR,
        is_trainable=False,
        device_map=None,
    )

    model.to(device)
    model.eval()

    print("[INFO] Loading processor (tokenizer + vision + chat template)...")
    # Use LORA_DIR so we pick up your chat_template.jinja if saved there
    processor = AutoProcessor.from_pretrained(
        LORA_DIR,
        trust_remote_code=True,
    )

    return model, processor, device


# -------------------------
# Inference on two frames
# -------------------------
@torch.no_grad()
def predict_reward_from_two_frames(
    model,
    processor,
    device: str,
    prev_pil: Image.Image,
    curr_pil: Image.Image,
    max_new_tokens: int = 32,
):
    """
    Use fine-tuned Qwen3-VL + LoRA to predict reward from 2 frames.
    Mirrors your new dataset messages, but uses:
      1) apply_chat_template -> text prompt
      2) processor(text, images=...) -> model inputs (dict of tensors)
    """
    # 1) Build messages exactly like in your new dataset (but with PIL images)
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a VLM that takes 2 consecutive frames from the 'Pusher-V5' environment and generate rewards base on them.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "<image_1>"},
                {"type": "image", "image": "<image_2>"},
                {
                    "type": "text",
                    "text": (
                        "The first image shows the scene at the previous time step; "
                        "the second image shows the scene at the current time step. "
                        "From these two images, estimate the immediate reward at the "
                        "current time step. Please output a single float number."
                    ),
                },
            ],
        },
    ]

    # 2) Turn messages into a *text* prompt using the chat template
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,              # <- IMPORTANT: we want a string, not tensors
        add_generation_prompt=True,
    )

    # 3) Build model inputs from text + images
    #    Qwen3-VL processor handles multi-image inputs like this:
    inputs = processor(
        text=prompt_text,
        images=[prev_pil, curr_pil],  # 2 consecutive frames
        return_tensors="pt",
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 4) Generate
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        use_cache=True,
    )

    # 5) Decode ONLY the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_len:]

    raw_text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    val = parse_reward(raw_text)
    return raw_text, val


# -------------------------
# Main rollout loop
# -------------------------
def main():
    model, processor, device = load_model_and_processor()

    print(f"[ENV] Creating env: {ENV_ID}")
    env = gym.make(ENV_ID, render_mode="rgb_array")

    all_true = []
    all_pred = []
    samples_used = 0

    for ep in range(NUM_EPISODES):
        obs, info = env.reset()
        frame_prev = env.render()
        step = 0

        print(f"\n=== Episode {ep} ===")

        while step < MAX_STEPS_PER_EP:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            frame_curr = env.render()
            step += 1

            prev_pil = pil_from_array(frame_prev)
            curr_pil = pil_from_array(frame_curr)

            raw, pred_reward = predict_reward_from_two_frames(
                model, processor, device, prev_pil, curr_pil
            )

            if pred_reward is not None:
                all_true.append(float(reward))
                all_pred.append(float(pred_reward))
                samples_used += 1
                print(
                    f"step={step:3d} | true={reward: .4f} | pred={pred_reward: .4f} | raw='{raw}'"
                )
            else:
                print(
                    f"step={step:3d} | true={reward: .4f} | pred=None | raw='{raw}'"
                )

            frame_prev = frame_curr

            if MAX_SAMPLES is not None and samples_used >= MAX_SAMPLES:
                break

            if terminated or truncated:
                break

        if MAX_SAMPLES is not None and samples_used >= MAX_SAMPLES:
            break

    env.close()

    if not all_true:
        print("\n[RESULT] No valid scalar predictions parsed. "
              "Check raw outputs above.")
        return

    y_true = np.array(all_true, dtype=np.float64)
    y_pred = np.array(all_pred, dtype=np.float64)

    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    corr = float("nan")
    if len(y_true) > 1:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])

    print("\n========= SUMMARY (env rollouts) =========")
    print(f"# samples (scalar preds): {len(y_true)}")
    print(f"MSE:  {mse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"Corr: {corr:.4f}")
    print("==========================================")


if __name__ == "__main__":
    main()