#!/usr/bin/env python3
import os
import re
import csv
from pathlib import Path

import torch
from PIL import Image

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

# ================== CONFIG ==================
FRAMES_DIR = "expert_frames"               # folder with frame_00000.png, frame_00001.png, ...
OUTPUT_CSV = "predicted_rewards.csv"       # CSV: index, prev_frame, curr_frame, predicted_reward
OUTPUT_TXT = "predicted_rewards.txt"       # TXT: one reward per line

BASE_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
LORA_PATH = "qwen3_vl_pusher_lora"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Optional: HF token if needed (or rely on `huggingface-cli login`)
HF_TOKEN = os.getenv("HF_TOKEN", None)
# ===========================================


def natural_key(s: str):
    """Sort strings with embedded numbers: frame_1, frame_2, ..., frame_10."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", s)]


def list_image_files(folder: str):
    """Return sorted list of image file paths from the folder."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = [f for f in os.listdir(folder) if Path(f).suffix.lower() in exts]
    files.sort(key=natural_key)
    return [os.path.join(folder, f) for f in files]


def extract_first_float(text: str) -> float:
    """Extract the first float from model output."""
    match = re.search(r"-?\d+(\.\d+)?", text)
    if not match:
        raise ValueError(f"Could not find a numeric reward in model output: {text!r}")
    return float(match.group(0))


def load_model_and_processor():
    print(f"Loading processor from: {BASE_MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    print(f"Loading base Qwen3-VL model from: {BASE_MODEL_NAME}")
    # IMPORTANT: no device_map="auto" here
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=DTYPE,
        device_map=None,
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    print(f"Loading LoRA adapter from: {LORA_PATH}")
    # Load LoRA weights onto the plain base_model (no offload, no device_map)
    model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
        is_trainable=False,
    )

    model.to(DEVICE)
    model.eval()
    return model, processor


def build_messages():
    """
    Match your fine-tuning format:

    {"messages": [
      {"role": "system", "content":[{"type":"text","text": "..."}]},
      {"role": "user",   "content":[
           {"type":"image"},
           {"type":"image"},
           {"type":"text","text": "..."}
      ]}
    ]}
    """
    system_text = (
        "You are a VLM that takes 2 consecutive frames from the 'Pusher-V5' "
        "environment and generate rewards base on them."
    )
    user_instruction = (
        "The first image shows the scene at the previous time step; "
        "the second image shows the scene at the current time step. "
        "From these two images, estimate the immediate reward at the current time step. "
        "Please output a single float number."
    )

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_text}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},  # first frame (previous timestep)
                {"type": "image"},  # second frame (current timestep)
                {"type": "text", "text": user_instruction},
            ],
        },
    ]
    return messages


def predict_reward_for_pair(model, processor, img_prev_path: str, img_curr_path: str) -> float:
    img_prev = Image.open(img_prev_path).convert("RGB")
    img_curr = Image.open(img_curr_path).convert("RGB")

    messages = build_messages()

    # Turn chat messages into text prompt
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Prepare inputs (text + 2 images)
    inputs = processor(
        text=[chat_text],
        images=[img_prev, img_curr],
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )

    # Decode output text and extract numeric reward
    text_out = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()

    reward = extract_first_float(text_out)
    return reward


def main():
    # 1) Collect frames
    frame_paths = list_image_files(FRAMES_DIR)
    if len(frame_paths) < 2:
        raise RuntimeError(f"Need at least 2 images in {FRAMES_DIR}, found {len(frame_paths)}")

    print(f"Found {len(frame_paths)} frames in '{FRAMES_DIR}'")

    # 2) Load model + processor
    model, processor = load_model_and_processor()

    # 3) Predict rewards for consecutive frame pairs
    rows = []
    rewards_only = []
    for i in range(len(frame_paths) - 1):
        img_prev = frame_paths[i]
        img_curr = frame_paths[i + 1]

        print(f"[{i}] {os.path.basename(img_prev)} -> {os.path.basename(img_curr)}")
        reward = predict_reward_for_pair(model, processor, img_prev, img_curr)
        print(f"    Predicted reward: {reward:.6f}")

        rows.append({
            "index": i,
            "prev_frame": os.path.basename(img_prev),
            "curr_frame": os.path.basename(img_curr),
            "predicted_reward": reward,
        })
        rewards_only.append(reward)

    # 4) Save CSV
    print(f"Writing predictions to {OUTPUT_CSV}")
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["index", "prev_frame", "curr_frame", "predicted_reward"]
        )
        writer.writeheader()
        writer.writerows(rows)

    # 5) Save TXT (just rewards)
    print(f"Writing rewards to {OUTPUT_TXT}")
    with open(OUTPUT_TXT, "w") as f:
        for r in rewards_only:
            f.write(f"{r}\n")

    print("Done.")


if __name__ == "__main__":
    main()
