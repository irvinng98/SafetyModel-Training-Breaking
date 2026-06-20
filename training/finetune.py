import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_from_disk

# 4-bit quantization config — fits in 8GB VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# LoRA config — low rank keeps VRAM usage down
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["qkv_proj", "o_proj"],  # Phi-3 uses fused QKV
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # should be ~1-2% of total params

# Load data
train_data = load_from_disk("data/train")
val_data = load_from_disk("data/val")

# Training args — conservative for 8GB VRAM
training_args = TrainingArguments(
    output_dir="models/phi3-safety-qlora",
    num_train_epochs=3,
    per_device_train_batch_size=2,   # increase to 4 if no OOM
    gradient_accumulation_steps=8,   # effective batch size = 16
    warmup_steps=50,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=25,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    report_to="wandb",               # free experiment tracking
    run_name="phi3-safety-qlora-v1",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    dataset_text_field="text",
    max_seq_length=512,
)

trainer.train()
trainer.save_model("models/phi3-safety-qlora/final")
print("Fine-tuning complete.")