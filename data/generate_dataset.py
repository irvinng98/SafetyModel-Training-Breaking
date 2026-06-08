# Here, we use our local Ollama model to generate example prompts based on the safe/unsafe categories from our taxonomy

import json                                                                                             # for parsing and saving JSON data
import requests                                                                                      # for making HTTP calls to Ollama's local server
import random                                                                                       
from taxonomy import UNSAFE_CATEGORIES, SAFE_CATEGORIES       # pulls in the category list defined in taxonomy.py
from tqdm import tqdm                                                                        # For progress/status bar

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
def generate_unsafe_example(category):
    prompt = f"""Generate a realistic harmful user request in the category of '{category}'. 
    Then write TWO responses: 
    1) UNSAFE_RESPONSE: A response that complies with the unsafe request
    2) SAFE_RESPONSE: A response that firmly but politely refuses the unsafe request
    
    Format as JSON:
    {{"instruction": "...", "unsafe_response": "...", "safe_response":"..."}}
    Output JSON only.
    """
    raw = ollama_generate(prompt)
    try:
        return json.loads(raw.strip())  # The try/except handles the case where Phi-3 returns malformed JSON — which will happen fairly often with smaller models. When that happens it just returns None and the example gets silently skipped downstream.
    except:
        return None


# Here, we generate safe prompts. safe_response will be a normal helpful answer and unsafe_response is null and we drop the field entirely.
def generate_safe_example(category):
    prompt = f"""Generate a realistic benign user request in the category of '{category}'.
    Then write a helpful, appropriate response.
    
    Format as JSON:
    {{"instruction": "...", "safe_response": "..."}}
    Output JSON only.
    """
    raw = ollama_generate(prompt)
    try:
        return json.loads(raw.strip())
    except:
        return None

# Dataset generation loop. Here Ollama iterates over every category and generates 200 examples each
dataset = [ ]
for category in UNSAFE_CATEGORIES:
    print(f"\n[UNSAFE] Generating examples for category: '{category}'")
    for _ in tqdm(range(200), desc=category):                       # The _ is conventional Python for "I don't need this loop variable" — it's just a counter.
        example = generate_unsafe_example(category)          # Generates the output JSON for each example prompt where each gets a dictionary: instruction, unsafe_response, safe_response
        if example:                                                                    # After generating an example prompt, two fields are added to each individual example dictionary that gets returned by generate_unsafe_example()
            example["category"] = category                              # Adds category to the dataset JSON for each example
            example["label"] = "unsafe"                                     # Adds label to the dataset JSON for each example. In this case, because they're all unsafe prompts, all will be "unsafe"
            dataset.append(example)                                        # Adds the final JSON to the dataset. Each JSON dictionary will have instruction, unsafe_response, safe_response, category, and label

for category in SAFE_CATEGORIES:
    print(f"\n[SAFE] Generating examples for category: '{category}'")
    for _ in tqdm(range(200), desc=category):
        example = generate_safe_example(category)
        if example:
            example["category"] = category
            example["label"] = "safe"
            example["unsafe_response"] = None  # explicit null for schema consistency
            dataset.append(example)  

# Write the whole dataset to a single JSON file with readable indentation. This is your raw data before any train/val/test splitting happens in the next script.       
with open("data/raw_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"\n Done. Generated {len(dataset)} valid examples out of {(len(UNSAFE_CATEGORIES) + len(SAFE_CATEGORIES)) * 200} attempts.")