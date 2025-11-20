import os
import math
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
from tqdm.auto import tqdm

# ============================
# CONFIG – edit these if needed
# ============================

# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"   # HF id or local folder
MODEL_ID = r"models/tinyllama-1.1b-chat"
# DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_ID = "zwhe99/DeepMath-103K"
# TRAIN_SPLIT = "train_sft[:1%]"                    # keep small while debugging
TRAIN_SPLIT = "train[:5%]"                  # use full train split for real training
# OUTPUT_DIR = "outputs/mistral7b_lora_ultrachat_manual"
OUTPUT_DIR = "outputs/tinyllama_lora_deepmath_manual"

USE_LORA = True          # False -> full finetune (all weights)
MAX_SEQ_LEN = 1024       # lower if you hit OOM
EPOCHS = 1
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

SYSTEM_PROMPT = "You are a helpful, concise assistant."


def get_device():
    # Print what is being returned
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Selected device: {device}")
    return device


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_and_format_dataset(tokenizer):
    """
    Load DeepMath-103K and turn (question, solution) into a chat-formatted text.
    We use the r1_solution_* fields if available; fallback to final_answer.
    """
    ds = load_dataset(DATASET_ID, split=TRAIN_SPLIT)

    def format_example(example):
        question = example.get("question", "")

        # Prefer the worked solution if present
        solution = ""
        for key in ["r1_solution_1", "r1_solution_2", "r1_solution_3"]:
            if key in example and example[key] and str(example[key]).strip():
                solution = str(example[key])
                break

        # Fallback: just the final answer if no solution text is available
        if not solution.strip():
            solution = str(example.get("final_answer", ""))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        if solution.strip():
            messages.append({"role": "assistant", "content": solution})

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    ds = ds.map(
        format_example,
        remove_columns=[c for c in ds.column_names if c not in ["text"]],
    )
    return ds



def tokenize_dataset(tokenizer, text_ds):
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )

    tokenized = text_ds.map(
        tokenize_batch,
        batched=True,
        remove_columns=text_ds.column_names,
    )
    return tokenized


def load_model(device):
    # bf16 if possible, otherwise fp16 or fp32
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    """model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )"""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,  # Will soon be deprecated in favor of bitsandbytes_config
        device_map="auto",
        local_files_only=True,  # <- this is important
    )
    # model.to(device)
    model.config.use_cache = False
    return model


def wrap_with_lora(model):
    peft_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_cfg)
    return model


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = get_tokenizer()

    print(f"Loading dataset: {DATASET_ID} / {TRAIN_SPLIT}")
    text_ds = load_and_format_dataset(tokenizer)
    print("Train samples:", len(text_ds))

    print("Tokenizing dataset...")
    tokenized_ds = tokenize_dataset(tokenizer, text_ds)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # for causal LM
    )

    train_loader = DataLoader(
        tokenized_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
    )

    print("Loading base model:", MODEL_ID)
    model = load_model(device)

    if USE_LORA:
        print("Applying LoRA...")
        model = wrap_with_lora(model)
        model.print_trainable_parameters()

    model.train()

    # Only train parameters that require grad (LoRA params if USE_LORA=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)/1e6:.2f}M")

    optimizer = AdamW(trainable_params, lr=LEARNING_RATE)

    # Simple linear warmup + decay scheduler
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        running_loss = 0.0

        for batch in progress:
            # batch has input_ids, attention_mask, labels
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],  # LM head will shift and compute CE loss
            )
            loss = outputs.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            running_loss += loss.item()
            avg_loss = running_loss / global_step
            progress.set_postfix(loss=loss.item(), avg_loss=avg_loss)

    print("Training finished. Saving LoRA model + tokenizer to:", OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
