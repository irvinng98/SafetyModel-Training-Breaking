# Agentic Red-Teaming of Safety-Fine-Tuned LLMs

An end-to-end pipeline for fine-tuning a small language model (Phi-3-mini) to refuse harmful requests using QLoRA, then autonomously red-teaming it with a LangGraph agent to empirically measure how much that safety training actually holds under adversarial pressure.

## Project structure

```
SafetyModel-Training-Breaking/
├── data/
│   ├── taxonomy.py           # Unsafe/safe category definitions
│   ├── generate_dataset.py   # Synthetic data generation via Ollama
│   ├── format_dataset.py     # Formats data for instruction tuning
│   └── raw_dataset.json      # Generated output (git-ignored)
├── training/
│   └── finetune.py           # QLoRA fine-tuning with PEFT + TRL
├── redteam/
│   ├── agent.py              # LangGraph red-team agent
│   └── run_redteam.py        # Runs agent across all categories
├── evaluation/
│   └── baseline_eval.py      # Refusal rate comparison: base vs fine-tuned
├── analysis/
│   └── analyze_results.py    # Metrics, plots, summary stats
├── results/                  # CSVs and charts (git-ignored)
├── models/                   # Saved adapters (git-ignored)
├── requirements.txt
└── README.md
```

## Hardware requirements

| Component | Minimum | Notes |
|---|---|---|
| GPU VRAM | 8 GB | Tested on RTX 3060 Ti. Lower VRAM requires reducing batch size further. |
| RAM | 16 GB | 32 GB recommended for comfortable data processing |
| Disk | 30 GB free | Models + dataset + results |
| CUDA | 12.1+ | Required for bitsandbytes 4-bit quantization |

## Installation

### 1. Create and activate a conda environment

```bash
conda create -n redteam python=3.11
conda activate redteam
```

### 2. Install PyTorch with CUDA support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install ML dependencies

```bash
pip install transformers==4.44.0
pip install peft==0.12.0
pip install trl==0.11.0
pip install bitsandbytes==0.43.3
pip install accelerate==0.34.0
```

### 4. Install agentic and utility dependencies

```bash
pip install langchain==0.3.0 langgraph==0.2.0 langchain-community
pip install datasets scikit-learn pandas numpy matplotlib seaborn
pip install wandb jupyter
```

### 5. Install Ollama

Download from [ollama.com](https://ollama.com), then pull the models used for data generation and red-teaming:

```bash
ollama pull phi3
ollama pull llama3.2
```

### 6. Verify GPU access

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1e9)  # should show ~8.0
```

## Usage

### Step 1 — Generate the synthetic dataset

This uses your local Phi-3 via Ollama to produce ~1000 safe/unsafe instruction pairs across five harm categories. No external API required.

```bash
python data/generate_dataset.py
python data/format_dataset.py
```

Output: `data/raw_dataset.json` and HuggingFace dataset splits saved to `data/train`, `data/val`, `data/test`.

### Step 2 — Fine-tune Phi-3-mini with QLoRA

```bash
python training/finetune.py
```

This trains a LoRA adapter on top of the quantized base model. Training runs for 3 epochs with an effective batch size of 16 via gradient accumulation. The final adapter is saved to `models/phi3-safety-qlora/final`.

Expected VRAM usage: ~6–7 GB. If you hit OOM errors, reduce `per_device_train_batch_size` to 1.

### Step 3 — Run baseline evaluation

Measures refusal rates on the held-out test set for both the base model and the fine-tuned model. Models are loaded sequentially to stay within 8 GB VRAM.

```bash
python evaluation/baseline_eval.py
```

Output: `results/baseline_eval.csv`

### Step 4 — Run the red-team agent

Runs the LangGraph red-team loop across all five unsafe categories. The attacker LLM (llama3.2 via Ollama) generates adversarial prompts, tests them against the fine-tuned model, classifies compliance vs refusal, and updates its strategy over 20 iterations per category.

```bash
python redteam/run_redteam.py
```

Output: `results/redteam_results.csv`, `results/redteam_summary.csv`

### Step 5 — Analyze results

```bash
python analysis/analyze_results.py
```

Produces attack success rate by category, success rate over iterations, and a comparison of base vs fine-tuned refusal rates.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `WANDB_API_KEY` | Recommended | Weights & Biases API key for experiment tracking. Get one free at [wandb.ai](https://wandb.ai). Set to `WANDB_MODE=disabled` to skip tracking entirely. |
| `WANDB_PROJECT` | No | W&B project name. Defaults to the `run_name` set in `TrainingArguments`. |
| `HF_TOKEN` | No | HuggingFace token. Only needed if `microsoft/Phi-3-mini-4k-instruct` is gated on your account. |
| `OLLAMA_HOST` | No | Override the Ollama server address. Defaults to `http://localhost:11434`. Useful if running Ollama on a separate machine. |
| `CUDA_VISIBLE_DEVICES` | No | Pin training to a specific GPU, e.g. `CUDA_VISIBLE_DEVICES=0`. |

Set these in your shell or in a `.env` file:

```bash
export WANDB_API_KEY=your_key_here
export HF_TOKEN=your_token_here
```

## Key design decisions

**Why Phi-3-mini?** At 3.8B parameters it fits comfortably in 8 GB VRAM under 4-bit quantization, leaving headroom for the LoRA adapter and optimizer states. Larger models (7B+) require either more VRAM or aggressive offloading that slows training significantly.

**Why local Ollama for data generation and red-teaming?** Keeps the entire pipeline offline and free. No OpenAI or Anthropic API costs.

**Why keyword-based refusal classification in `baseline_eval.py`?** Speed. The LLM-based classifier in the red-team agent is more accurate but adds latency per call. Baseline eval runs over the full test set, so the lightweight classifier is a practical trade-off. You can swap in the LLM classifier for higher-fidelity results if needed.

## License

MIT