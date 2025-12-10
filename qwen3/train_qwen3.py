#!/usr/bin/env python3
import os
from datasets import load_dataset
import torch

from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig

# ----------------------
# Basic config
# ----------------------
MODEL_ID = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
DATA_PATH = "pusher_vlm_data/qwen3_vl_multi_task.jsonl"
OUTPUT_DIR = "qwen3_vl_pusher_lora"

MAX_SEQ_LEN = 2048  # text tokens; Qwen3-VL supports long context, but this is enough
BATCH_SIZE = 2      # start small; increase if VRAM allows
GRAD_ACCUM = 4      # effective batch size = BATCH_SIZE * GRAD_ACCUM
LR = 2e-4
EPOCHS = 1          # bump up once it works

# Optional for better CUDA memory behavior
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    # ----------------------
    # 1) Load model via Unsloth
    # ----------------------
    print("Loading Qwen3-VL-8B-Instruct (Unsloth 4-bit)...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = MODEL_ID,
        load_in_4bit = True,                  # QLoRA-style 4bit
        use_gradient_checkpointing = "unsloth",
        trust_remote_code = True,
    )

    # Add LoRA adapters – tune both vision + language parts
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True,
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0.0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
        target_modules = "all-linear",
        modules_to_save = ["lm_head", "embed_tokens"],
    )

    # ----------------------
    # 2) Load dataset
    # ----------------------
    print("Loading dataset...")
    ds = load_dataset(
        "json",
        data_files={"train": DATA_PATH},
        split="train",
    )

    # Quick tiny validation split
    ds_split = ds.train_test_split(test_size=0.01, seed=42)
    train_ds = ds_split["train"]
    eval_ds  = ds_split["test"]

    # NOTE:
    # Unsloth vision fine-tuning expects each example to have a `messages`
    # field in the exact format we created above.
    # The patched SFTTrainer uses the chat template + vision processor
    # under the hood when given FastVisionModel. :contentReference[oaicite:1]{index=1}

    # ----------------------
    # 2.5) Define formatting_func for Unsloth / TRL
    # ----------------------
    # Unsloth now *requires* a formatting_func when the dataset uses `messages`.
    # This function:
    #   - works for both single-example and batched calls
    #   - converts each `messages` conversation into a flat chat-template string
    #
    def formatting_func(batch):
        """
        batch: either
          - a single example: {"messages": [ {...}, {...}, ... ]}
          - or a dict of lists: {"messages": [ [ {...}, ... ], [ {...}, ... ], ... ]}
        We normalize to a list of conversations, then apply the chat template.
        """
        messages_field = batch["messages"]

        # Case 1: single example => messages_field is a list[dict]
        if isinstance(messages_field, list) and len(messages_field) > 0 and isinstance(messages_field[0], dict):
            conversations = [messages_field]
        else:
            # Case 2: batched => list[list[dict]]
            conversations = messages_field

        texts = [
            tokenizer.apply_chat_template(
                conv,
                tokenize = False,
                add_generation_prompt = False,
            )
            for conv in conversations
        ]
        # Unsloth / TRL expect a list of strings when batched=True
        return texts

    # ----------------------
    # 3) Setup SFTTrainer
    # ----------------------
    sft_config = SFTConfig(
        output_dir = OUTPUT_DIR,
        num_train_epochs = EPOCHS,
        learning_rate = LR,
        lr_scheduler_type = "cosine",
        warmup_ratio = 0.03,

        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        per_device_eval_batch_size = 1,

        logging_steps = 10,
        eval_strategy = "steps",
        eval_steps = 1000,
        save_strategy = "steps",
        save_steps = 1000,
        save_total_limit = 3,

        bf16 = torch.cuda.is_available(),   # or fp16 if your GPU needs it
        fp16 = False,
        gradient_checkpointing = True,
        max_seq_length = MAX_SEQ_LEN,
        packing = False,     # keep one conversation per sample
        report_to = "none",
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_ds,
        eval_dataset = eval_ds,
        # We no longer set dataset_text_field="messages" here;
        # instead, we tell Unsloth/TRl how to turn `messages` into text.
        formatting_func = formatting_func,
        args = sft_config,
    )

    print("Starting training...")
    trainer.train()

    print("Saving LoRA adapters...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save LoRA (or PEFT) weights + config into OUTPUT_DIR
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Done. LoRA adapters + tokenizer saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
