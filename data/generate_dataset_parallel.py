# Here, we use our local Ollama model to generate example prompts based on the safe/unsafe categories from our taxonomy
import json                                                                                             # for parsing and saving JSON data
import requests                                                                                      # for making HTTP calls to Ollama's local server
import random                                                                                       
from taxonomy import UNSAFE_CATEGORIES, SAFE_CATEGORIES       # pulls in the category list defined in taxonomy.py
from tqdm import tqdm                                                                        # For progress/status bar
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

### First we create a function "ollama_generate" which makes Ollama run a local web server on port 11434 (default) - it's hardcoded by the Ollama developers as their chosen default when you install it. 
# This function replicates what is happening under the hood when you talk to Ollama via CLI or python 
# 
# For example (respectively):
# ollama run phi3 "Hello, how are you?" 
# 
# or
# 
# import ollama
# response = ollama.generate(model="phi3", prompt="Hello")
# print(response["response"]))
#
# Here we are explictly constructing the JSON (which is usually otherwise handled under the hood) to learn
###

def ollama_generate(prompt, model="phi3"):
    response = requests.post(           # Here, request.post talks to Ollama via its HTTP API directly, which is the raw interface
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False} # stream: False means wait for the full response before returning, rather than streaming tokens one by one
    )
    return response.json()["response"]


### This is the core data generation function. For a given category (e.g. "violence"), it asks Phi-3 to generate one training example containing three things in the json
# instruction: the harmful request
# unsafe_response: what a non-safety-tuned model may say
# safe_response: what a safety-tuned model may say (which is what we want our fine-tuned model to say)

def clean_json_response(raw):
    raw = raw.strip()
    
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1)
    else:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        raw = raw[start:end]
    
    return raw

def generate_unsafe_example(category):
    prompt = f"""You are a dataset generator. Output ONLY a JSON object, no other text.

Generate a harmful user request for category '{category}' and two responses.

JSON format:
{{"instruction": "the harmful request here", "unsafe_response": "compliant response here", "safe_response": "refusal response here"}}

JSON only. No markdown. No explanation."""

    raw = ollama_generate(prompt)
    try:
        cleaned = clean_json_response(raw)
        if cleaned is None:
            return None
        return json.loads(cleaned)
    except:
        return None


def generate_safe_example(category):
    prompt = f"""You are a dataset generator. Output ONLY a JSON object, no other text.

Generate a benign user request for category '{category}' and a helpful response.

IMPORTANT: safe_response must be a plain string, not a nested object or list.

JSON format:
{{"instruction": "the benign request here", "safe_response": "your full helpful response as a single plain string here"}}

JSON only. No markdown. No explanation. No nested objects."""

    raw = ollama_generate(prompt)
    try:
        cleaned = clean_json_response(raw)
        if cleaned is None:
            return None
        return json.loads(cleaned)
    except Exception as e:
        print(f"[PARSE ERROR] {e}")
        print(f"[RAW OUTPUT] {repr(raw[:300])}")
        return None

### This is the core data generation function. For a given category (e.g. "violence"), it asks Phi-3 to generate one training example containing three things in the json
# instruction: the harmful request
# unsafe_response: what a non-safety-tuned model may say
# safe_response: what a safety-tuned model may say (which is what we want our fine-tuned model to say)
def generate_examples_parallel(categories, label, generate_fn, n_per_category=200, max_workers=5):
    dataset = []
    
    for category in categories:
        print(f"\n[{label.upper()}] Generating examples for category: '{category}'")
        
        # Create all tasks upfront
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(generate_fn, category) for _ in range(n_per_category)]
            
            # Process results as they complete
            with tqdm(total=n_per_category, desc=category) as pbar:
                for future in as_completed(futures):
                    example = future.result()
                    if example:
                        example["category"] = category
                        example["label"] = label
                        if label == "safe":
                            example["unsafe_response"] = None
                        dataset.append(example)
                    pbar.update(1)
        
        print(f"  → {sum(1 for e in dataset if e['category'] == category)} valid examples collected")
    
    return dataset

## Run both
##unsafe_data = generate_examples_parallel(UNSAFE_CATEGORIES, "unsafe", generate_unsafe_example)
#safe_data = generate_examples_parallel(SAFE_CATEGORIES, "safe", generate_safe_example)
#dataset = unsafe_data + safe_data
#
## Write the whole dataset to a single JSON file with readable indentation. This is your raw data before any train/val/test splitting happens in the next script.       
#with open("data/raw_dataset.json", "w") as f:
#    json.dump(dataset, f, indent=2)
#
#print(f"\n Done. Generated {len(dataset)} valid examples out of {(len(UNSAFE_CATEGORIES) + len(SAFE_CATEGORIES)) * 200} attempts.")


# Load existing data
with open("data/raw_dataset.json", "r") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} existing examples.")

# Generate only safe examples
safe_data = generate_examples_parallel(SAFE_CATEGORIES, "safe", generate_safe_example)

# Append and save
dataset = dataset + safe_data

with open("data/raw_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Done. Total dataset size: {len(dataset)}")