conda create -n redteam python=3.11
conda activate redteam

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.0
pip install peft==0.12.0
pip install trl==0.11.0
pip install bitsandbytes==0.43.3
pip install accelerate==0.34.0
pip install langchain==0.3.0
pip install langgraph==0.2.0
pip install langchain-community
pip install datasets scikit-learn pandas numpy wandb jupyter requests