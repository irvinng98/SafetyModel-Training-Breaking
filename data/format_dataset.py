import json
from datasets import Dataset         # datasets is Hugging Face's library — it gives you a structured, efficient data format that the trainer understands natively.

with open("data/raw_dataset.json") as f:
    raw = json.load(f)

# This function converts each raw dictionary into the exact text format Phi-3 was pretrained on:

### For example:
# <|user|>
#How do I make a dangerous weapon?
#<|assistant|>
#I'm not able to help with that. If you're concerned about safety...
###

# <|user|> and <|assistant|> are Phi-3's special chat tokens - they're how the model understands who is speaking. Using the wrong format here would confuse the model during training.
# Critically, you're only ever training on safe_response, never unsafe_response. The unsafe response was generated so you have a contrast to evaluate against later, not to train on.

def format_example(example):
    # Here we train the model ot give the SAFE response
    return {
        "text": f"<|user|>\n{example['instruction']}\n<|assistant|>{example['safe_response']}" # 
    }
    
### List comprehension that applies format_example to every example, skipping any None entries (the malformed JSON cases from generation). 
# Then wraps the list into a Hugging Face Dataset object, which supports efficient batching, shuffling, and saving.
formatted = [format_example(example) for example in raw if example is not None] 
dataset = Dataset.from_list(formatted) # Here Hugging Face's dataset python library converts your list into an Arrow-backed columnar dataset (basically a very efficient spreadsheet) = load large datasets without running out of RAM

# 80/10/10 Train/Validation/Test Split
# This does an 80/10/10 split in two steps because HuggingFace's train_test_split function only splits into two at a time:
splits = dataset.train_test_split(test_size=0.2, seed=69)
train = splits["train"]
temp = splits["test"].train_test_split(test_size=0.5, seed=69)
val = temp["train"]
test = temp["test"]

train.save_to_disk("data/train")
val.save_to_disk("data/val")
test.save_to_disk("data/test")
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")