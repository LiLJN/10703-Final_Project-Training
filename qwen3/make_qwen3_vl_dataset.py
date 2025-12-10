import json
from pathlib import Path

RAW_PATH = Path("pusher_vlm_data/meta.jsonl")
OUT_PATH = Path("pusher_vlm_data/qwen3_vl_multi_task.jsonl")
IMAGE_ROOT = Path("pusher_vlm_data/images")

def fmt(vec, precision=4):
    return "[" + ", ".join(f"{x:.{precision}f}" for x in vec) + "]"

def make_image_path(filename: str) -> str:
    """Return path string that Qwen3-VL will load (relative or absolute)."""
    return str(IMAGE_ROOT / filename)

SYSTEM_MSG = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": (
                "You are a VLM that takes 2 consecutive frames from the 'Pusher-V5' environment and generate rewards base on them."
            ),
        }
    ],
}

# ---------- 1) Reward example: needs TWO images (prev + curr), NO numeric info leaked ----------

def make_reward_example(rec):
    rew = rec["reward"]

    user_text = (
        "The first image shows the scene at the previous time step; "
        "the second image shows the scene at the current time step. "
        "From these two images, estimate the immediate reward at the current time step. "
        "Please output a single float number."
    )

    assistant_text = f"{rew:.4f}"

    return {
        "messages": [
            SYSTEM_MSG,
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": make_image_path(rec["prev_image"]),
                    },
                    {
                        "type": "image",
                        "image": make_image_path(rec["curr_image"]),
                    },
                    {
                        "type": "text",
                        "text": user_text,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": assistant_text,
                    }
                ],
            },
        ]
    }


# ---------- 2) Fingertip example: ONE image (curr), no pose leaked in prompt ----------

# def make_fingertip_example(rec):
#     curr_f = fmt(rec["curr_fingertip_obs"])

#     user_text = (
#         "You are given a single camera image from a robot pushing task. "
#         "The image shows a robotic arm, an object on a table, and a goal location that the robot is trying to push the object toward. "
#         "From this image alone, estimate the 3D position of the robot fingertip in the world frame. "
#         "Question: What is the current fingertip position in 3D?\n"
#         "Please output a list of 3 floats like [x, y, z]."
#     )

#     assistant_text = curr_f

#     return {
#         "messages": [
#             SYSTEM_MSG,
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "image",
#                         "image": make_image_path(rec["curr_image"]),
#                     },
#                     {
#                         "type": "text",
#                         "text": user_text,
#                     },
#                 ],
#             },
#             {
#                 "role": "assistant",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": assistant_text,
#                     }
#                 ],
#             },
#         ]
#     }


# # ---------- 3) Object example: ONE image (curr), no object position leaked in prompt ----------

# def make_object_example(rec):
#     curr_o = fmt(rec["curr_object_obs"])

#     user_text = (
#         "You are given a single camera image from a robot pushing task. "
#         "The image shows a robotic arm and an object on a table that the robot is trying to push toward a goal. "
#         "From this image alone, estimate the 3D position of the object in the world frame. "
#         "Question: What is the current object position in 3D?\n"
#         "Please output a list of 3 floats like [x, y, z]."
#     )

#     assistant_text = curr_o

#     return {
#         "messages": [
#             SYSTEM_MSG,
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "image",
#                         "image": make_image_path(rec["curr_image"]),
#                     },
#                     {
#                         "type": "text",
#                         "text": user_text,
#                     },
#                 ],
#             },
#             {
#                 "role": "assistant",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": assistant_text,
#                     }
#                 ],
#             },
#         ]
#     }


# # ---------- 4) Goal example: ONE image (curr), no goal position leaked in prompt ----------

# def make_goal_example(rec):
#     curr_g = fmt(rec["curr_goal_obs"])

#     user_text = (
#         "You are given a single camera image from a robot pushing task. "
#         "The image shows a robotic arm, an object on a table, and a visual marker or region on the table that indicates the goal location. "
#         "From this image alone, estimate the 3D position of the goal in the world frame. "
#         "Question: What is the current goal position in 3D?\n"
#         "Please output a list of 3 floats like [x, y, z]."
#     )

#     assistant_text = curr_g

#     return {
#         "messages": [
#             SYSTEM_MSG,
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "image",
#                         "image": make_image_path(rec["curr_image"]),
#                     },
#                     {
#                         "type": "text",
#                         "text": user_text,
#                     },
#                 ],
#             },
#             {
#                 "role": "assistant",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": assistant_text,
#                     }
#                 ],
#             },
#         ]
#     }

# ---------- Main: turn raw transitions into multi-task VL chat data ----------

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_in, n_out = 0, 0

    with RAW_PATH.open("r") as fin, OUT_PATH.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)

            # 4 tasks per transition
            examples = [
                make_reward_example(rec),
                # make_fingertip_example(rec),
                # make_object_example(rec),
                # make_goal_example(rec),
            ]

            for ex in examples:
                fout.write(json.dumps(ex) + "\n")
                n_out += 1

    print(f"Read {n_in} transitions, wrote {n_out} training examples to {OUT_PATH}")


if __name__ == "__main__":
    main()