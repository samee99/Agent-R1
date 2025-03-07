<h1 align="center"> Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning </h1>

## Overview

Reinforcement learning has catalyzed the evolution of Large Language Models (LLMs) from simple chatbots to powerful reasoning engines capable of superhuman performance on complex tasks like mathematics and coding. Agent-R1 represents the next frontier in this progression: a framework for training agentic LLMs through end-to-end reinforcement learning.

Traditional approaches to building AI agents often rely on manually constructed operational graphs with language models positioned at specific decision nodes. While this approach can quickly produce prototypes, it struggles to handle the complexity and unpredictability of real-world scenarios. Agent-R1 takes a fundamentally different approach by training agents end-to-end on complex tasks, allowing them to develop flexible, adaptive strategies that would be impossible to script manually.

## Training Models

### Quick Start: Try Default Search Tool

#### Installation
```bash
# Clone the repository
git clone https://github.com/your-org/Agent-R1.git
cd Agent-R1

```

Install `verl`
```bash
# install verl together with some lightweight dependencies in setup.py
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.0.post2 --no-build-isolation
git clone https://github.com/volcengine/verl.git
cd verl
pip3 install -e .
```

Install FlagEmbedding and faiss-cpu
```bash
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding
pip3 install -e .
pip3 install faiss-cpu
```

#### 1. Download and preprocess HotpotQA dataset
```bash
# Create data directory
mkdir -p data/hotpotqa

# Run the preprocessing script
python examples/data_preprocess/hotpotqa.py --local_dir ./data/hotpotqa
```

This script will:
- Download the HotpotQA dataset directly from the source
- Process the data into the format required by Agent-R1
- Save the processed data as train.parquet and validation.parquet in the specified directory

#### 2. Download the search index and corpus
The search tool uses txtai embeddings to simulate internet searches using Wikipedia as a knowledge source.

```bash
# The search tool will automatically download the required index from the Hugging Face Hub
# You can verify the setup by running a simple test
python -c "from agent_r1.tool.tools.search_tool import SearchTool; tool = SearchTool(); print(tool.execute({'query': 'Who was Albert Einstein?'}))"
```

#### 3. Run PPO training with Qwen2.5-3B-Instruct
```bash
# Run the PPO training script
bash run_ppo.sh
```

This will start the training process using the Qwen2.5-3B-Instruct model. The training progress can be monitored through the console output and Weights & Biases dashboard.

#### 4. Monitor training
Training metrics are logged to Weights & Biases. You can view the training progress by visiting the URL provided in the console output.

#### 5. Use the trained model
After training, the model checkpoints will be saved in the `outputs` directory. You can use the trained model for inference or further fine-tuning.