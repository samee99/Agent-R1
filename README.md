<h1 align="center"> Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning </h1>

## Overview

**Reinforcement learning (RL)** has catalyzed the evolution of Large Language Models (LLMs) from simple **Chatbots (Level 1)** to powerful **Reasoners (Level 2)** capable of superhuman performance on complex tasks like mathematics and coding. Models trained with reinforcement learning have demonstrated remarkable abilities to develop complex reasoning strategies through exploration and exploitation, as seen in breakthrough models like DeepSeek's R1, which naturally learns to construct long reasoning chains to solve challenging problems.

The advancement of foundation models has fueled aspirations for true **Agents (Level 3)**.  Unlike chatbots and reasoners, agents not only utilize their internal knowledge but also actively explore external environments through autonomous action. Traditional agent approaches primarily rely on human-designed workflows, where models passively interact with environments according to predefined rules. While reasoners have freed us from the burden of prompt engineering through their ability to independently analyze and break down problems, a new question emerges: Can we enable models to independently take actions and explore environments on their own? The intersection of **RL & Agent** reveals this promising frontier—where models learn not just to reason but to act autonomously in complex, dynamic environments.

**Agent-R1** is an open-source framework designed to accelerate research and development at this critical intersection. Our framework employs **End-to-End** reinforcement learning to train agents in specific environments. Developers need only define domain-specific tools and reward functions to extend Agent-R1 to their unique use cases, eliminating the need for complex workflow engineering. We hope our modest contribution can benefit the open-source community, making it easier for researchers and developers to create and explore agents in their own domains, collectively advancing the development of autonomous agents.

## Algoritm
TODO:

## Key Features

- **Multi-turn Tool Calling**: End-to-end reinforcement learning on complete interaction trajectories, allowing agents to learn from sequences of actions
- **Custom Tools and Environments**: Compatible with mainstream LLM tool calling formats, making it easy to extend with your own tools and scenarios
- **Multiple RL Algorithms**: Supports diverse reinforcement learning approaches including PPO, GRPO, and REINFORCE++
- **Reasoning before Action**: Jointly optimize reasoning and action strategies over entire trajectories.

## Upcoming Features

- **Immediate Action Rewards**: Per-action reward mechanisms to complement trajectory-level reinforcement
- **Expanded Model Support**: Integration with more foundation models beyond the currently supported Qwen
- **Additional Use Cases**: More example implementations across diverse scenarios and domains

## Get Started

### Environment Setup

**Clone the repository**
```bash
git clone https://github.com/0russwest0/Agent-R1.git
cd Agent-R1
```

**Install `verl`**
```bash
mkdir -p envs
cd envs
conda create -n verl python==3.9
conda activate verl
# install verl together with some lightweight dependencies in setup.py
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
git clone https://github.com/volcengine/verl.git
cd verl
pip3 install -e .
```

### Quick Start: Try Default Search Tool on HotpotQA
#### 1. Install `FlagEmbedding` and `faiss`
```bash
pip3 install FlagEmbedding
pip3 install faiss-cpu
```

#### 2. Download and preprocess HotpotQA dataset
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

#### 3. Build hotpotqa search index
```bash
# Download the corpus file (gzipped)
mkdir -p data/corpus/hotpotqa
wget https://huggingface.co/datasets/BeIR/hotpotqa/resolve/main/corpus.jsonl.gz -O data/corpus/hotpotqa/corpus.jsonl.gz

# Extract the gzipped file
gunzip -c data/corpus/hotpotqa/corpus.jsonl.gz > data/corpus/hotpotqa/hpqa_corpus.jsonl

# Process the corpus and build the search index
python scripts/hotpotqa_search/process_hotpotqa.py
```

This script will:
- Load the corpus data
- Generate embeddings using the BAAI/bge-large-en-v1.5 model
- Build a FAISS index for efficient similarity search
- Save the embeddings and index files in the data/corpus/hotpotqa directory

#### 4. Run PPO/REINFORCE++/GRPO training with Qwen2.5-1.5B-Instruct
```bash
# Run the PPO training script
bash run_ppo.sh
# Run the REINFORCE++ training script
bash run_rpp.sh
# Run the GRPO training script
bash run_grpo.sh
```

This will start the training process using the Qwen2.5-1.5B-Instruct model. The training progress can be monitored through the console output and Weights & Biases dashboard.

## Extending Agent-R1 with Your Own Tools and Environments

Agent-R1 is designed to be easily extensible, allowing you to create custom tools and environments for your specific use cases. This section outlines the key files and components you need to modify or create.

### Key Components to Extend

1. **Custom Data Processing**
   - Create a new script in `examples/data_preprocess/` following `hotpotqa.py`
   - Implement data download functions (optional, see `download_file()` in `hotpotqa.py`)
   - Create data processing functions to transform raw data into the required format:
     - Define helper functions like `process_supporting_facts()` for specific data fields
     - Create a mapping function (`process_fn()`) to standardize each example
     - Format data with appropriate instruction templates
   - Save processed data as parquet files for training and validation

2. **Custom Tools**
   - Create a new Python file in `agent_r1/tool/tools/` (e.g., `my_custom_tool.py`)
   - Extend the `Tool` base class from `agent_r1.tool.tool_base`
   - Implement the required methods:
     - `__init__()`: Define tool name, description, and parameter schema
     - `execute()`: Implement the core functionality of your tool
     - `batch_execute()`: Implement batch processing capability if needed
   - Register your tool in `agent_r1/tool/tools/__init__.py` by adding it to the `_default_tools()` function

3. **Custom Reward Functions**
   - Create a new Python file in `verl/utils/reward_score/` following `qa_em_and_format.py`
   - Create specific scoring functions:
     - Format validation (see `compute_score_format()` which checks for proper output structure)
     - Answer evaluation (see `compute_score_answer()` which compares against ground truth)
     - Combined scoring functions (see `compute_score_format_answer()`)
   - Register your reward function in `verl/utils/reward_score/__init__.py`

### Example Workflow

To create a custom application with Agent-R1:

1. Identify the tools your agent will need to accomplish its tasks
2. Implement each tool by extending the `Tool` base class
3. Create appropriate data preprocessing for your specific use case:
   - Download and format your dataset
   - Define appropriate instruction templates
   - Structure data with necessary fields
4. Implement custom reward functions if needed:
   - Define how to extract answers from model outputs
   - Create scoring functions for format validation
   - Implement task-specific evaluation metrics
5. Configure a training script with appropriate parameters
6. Run the training script to train your agent

For detailed implementation guidance, examine the existing code:
- Tools: `agent_r1/tool/tools/calculator_tool.py`, `search_tool.py`
- Data processing: `examples/data_preprocess/hotpotqa.py`
- Reward functions: `verl/utils/reward_score/qa_em_and_format.py`