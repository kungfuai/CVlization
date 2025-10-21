## GPT-OSS Reinforcement Learning with GRPO

This example demonstrates how to train GPT-OSS 20B using GRPO (Group Relative Policy Optimization) with Unsloth for faster and more memory-efficient reinforcement learning.

### Features

- **GPT-OSS 20B**: OpenAI's open-source 20B parameter model
- **GRPO Training**: Reinforcement learning via reward optimization
- **3x Faster**: With Unsloth optimizations
- **50% Less VRAM**: Memory-efficient training on 15GB+ GPUs
- **Custom Rewards**: Modular reward functions for any task
- **Code Generation**: Example task with execution-based rewards

### What is GRPO?

GRPO (Group Relative Policy Optimization) is a reinforcement learning method that trains models to maximize rewards. Unlike supervised fine-tuning (SFT) where you provide explicit examples, GRPO:

1. **Generates multiple responses** per prompt
2. **Scores them with reward functions** (correctness, reasoning, etc.)
3. **Updates the model** to prefer high-reward responses

This is ideal for tasks like reasoning, math, code generation, and alignment.

### Prerequisites

1. NVIDIA GPU with at least 15GB VRAM (A10, A100, RTX 4090, etc.)
2. HuggingFace account and token (required for first-time model download)
3. Docker with NVIDIA runtime support

### Quickstart

1. (Optional) Set your HuggingFace token for first-time setup:
```bash
export HF_TOKEN=your_huggingface_token
```
*Note: Only required for initial model download. Subsequent runs use the cached model.*

2. Build the Docker image:
```bash
bash examples/text_gen/unsloth/gpt_oss_grpo/build.sh
```

3. Run training:
```bash
bash examples/text_gen/unsloth/gpt_oss_grpo/train.sh
```

### Training Details

- **Model**: unsloth/gpt-oss-20b (20 billion parameters)
- **RL Method**: GRPO with custom reward functions
- **Max Steps**: 10 (for quick testing)
- **Batch Size**: 1 per device
- **Learning Rate**: 5e-5
- **Beta (KL Penalty)**: 0.1
- **LoRA Rank**: 8 (optimized for 20B model)
- **Quantization**: 4-bit (bitsandbytes)
- **Context Length**: 512 tokens

Training takes approximately 10-15 minutes on an A10 GPU for 10 steps (~70 seconds per step).

### Current Dataset

The example uses a **code generation task** from the Unsloth notebook:

**Task**: Generate a fast matrix multiplication function in pure Python

**Prompt**:
```
Create a new fast matrix multiplication function using only native Python code.
You are given a list of list of numbers.
Output your new function in backticks using the format below:
```python
def matmul(A, B):
    return ...
```
```

This single prompt is replicated 1000 times in the dataset to provide enough training examples. The task tests the model's ability to generate correct, efficient code with proper formatting.

### Task Configuration

The default task is **code generation** (matrix multiplication) with the following reward components:

#### Reward Functions

All reward functions are defined in `reward_functions.py`:

1. **function_works** (score: -2.0 to 1.0)
   - Tests if generated code executes without errors
   - Returns 1.0 if function runs successfully
   - Returns -0.5 if execution fails
   - Returns -2.0 if code cannot be extracted or parsed

2. **no_cheating** (score: 1.0 or -20.0)
   - Heavily penalizes use of non-stdlib imports (numpy, scipy, etc.)
   - Returns 1.0 if only stdlib is used
   - Returns -20.0 if external libraries detected
   - Ensures pure Python implementation

3. **correctness_check** (score: -2.0 to 10.0)
   - Validates output against NumPy reference implementation
   - Tests with random matrices
   - Returns 10.0 if results match within tolerance
   - Returns -2.0 if results differ or execution fails

4. **speed_check** (score: varies)
   - Benchmarks execution time vs. NumPy baseline
   - Rewards implementations faster than naive Python
   - Uses cache thrashing and timeouts to prevent gaming
   - Score based on relative performance

### Customizing Rewards

Edit `reward_functions.py` to create custom reward functions for your task. GRPO reward functions have a different signature than typical reward functions:

```python
def my_custom_reward(completions, **kwargs) -> List[float]:
    """
    Custom reward function for GRPO.

    Args:
        completions: List of completion dictionaries with format:
                     [{"role": "assistant", "content": "..."}]
        **kwargs: Additional arguments (dataset, processing_class, etc.)

    Returns:
        List of reward scores (one per completion)
    """
    scores = []
    for completion in completions:
        # Extract the response text
        response = completion[0]["content"]

        # Your evaluation logic here
        score = evaluate_response(response)
        scores.append(score)

    return scores
```

Then add it to the trainer in `train.py`:

```python
from reward_functions import my_custom_reward

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        my_custom_reward,  # Add your function here
        # ... other reward functions
    ],
    args=training_args,
    train_dataset=dataset,
)
```

Note: Unlike some RL frameworks, GRPO does **not** use weighted reward combinations in the config. Each reward function returns its own score, and they are all logged separately.

### Configuration

All training is controlled via `config.yaml`. Key sections:

#### Model Configuration
```yaml
model:
  name: "unsloth/gpt-oss-20b"
  max_seq_length: 1024
  load_in_4bit: true
```

#### GRPO Settings
```yaml
grpo:
  beta: 0.1  # KL divergence penalty coefficient (prevents policy from deviating too far)
  temperature: 1.0  # Sampling diversity
  num_generations: 2  # Responses per prompt
  max_steps: 10
  learning_rate: 5.0e-5
  weight_decay: 0.01
```

#### Task & Prompts
```yaml
task:
  type: "code_generation"
  prompt: |
    Create a new fast matrix multiplication function using only native Python code.
    You are given a list of list of numbers.
    Output your new function in backticks using the format below:
    ```python
    def matmul(A, B):
        return ...
    ```
```

### Preparing Custom Data for RL Training

GRPO data preparation differs from supervised fine-tuning (SFT):

**SFT data format** (prompt-response pairs):
```json
{
  "prompt": [{"role": "user", "content": "What is 2+2?"}],
  "completion": [{"role": "assistant", "content": "2+2 equals 4."}]
}
```

**GRPO data format** (prompts + rewards, no gold answers):
```json
{
  "prompt": [{"role": "user", "content": "What is 2+2?"}],
  "answer": 0,
  "reasoning_effort": "low"
}
```

#### Key Differences

| Field | SFT | GRPO |
|-------|-----|------|
| `prompt` | User message | User message (same) |
| `completion` | Required - gold answer | **Not used** - model generates responses |
| `answer` | Not used | Placeholder (0) - for dataset compatibility |
| `reasoning_effort` | Not used | **Required** - "low", "medium", or "high" for GPT-OSS |

#### Creating Your Own GRPO Dataset

1. **Create a JSONL file** with prompts (no gold answers needed):

```json
{"prompt": [{"role": "user", "content": "Calculate 15 * 23"}], "answer": 0, "reasoning_effort": "low"}
{"prompt": [{"role": "user", "content": "What is the derivative of x^2?"}], "answer": 0, "reasoning_effort": "medium"}
{"prompt": [{"role": "user", "content": "Prove the Pythagorean theorem"}], "answer": 0, "reasoning_effort": "high"}
```

2. **Load in train.py**:

```python
from datasets import load_dataset

# Load from JSONL file
dataset = load_dataset('json', data_files='my_prompts.jsonl', split='train')
```

3. **Define reward functions** in `reward_functions.py`:

```python
def my_task_reward(completions, **kwargs) -> List[float]:
    """Score completions for your specific task."""
    rewards = []
    for completion in completions:
        response = completion[0]["content"]
        # Your scoring logic here
        score = evaluate_quality(response)
        rewards.append(score)
    return rewards
```

4. **Update `train.py`** to use your reward functions:

```python
from reward_functions import my_task_reward

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        my_task_reward,  # Your custom reward
        # Add other reward functions as needed
    ],
    args=training_args,
    train_dataset=dataset,
)
```

#### Reasoning Effort Guidelines

The `reasoning_effort` field tells GPT-OSS how much computation to allocate:

- **"low"**: Simple tasks (arithmetic, basic questions) - fast inference
- **"medium"**: Moderate complexity (multi-step problems, explanations)
- **"high"**: Complex reasoning (proofs, multi-hop logic, hard math)

For most tasks, start with `"low"` and increase if the task requires deeper thinking.

### Different Tasks

GRPO can be used for various tasks. Edit `config.yaml` to switch:

#### Code Generation
```yaml
task:
  type: "code"
  prompt_template: |
    Write a Python function that {task}
  problems:
    - "sorts a list of numbers"
    - "finds the factorial of n"
```

Create corresponding reward functions:
- Code executes without errors
- Passes test cases
- Follows style guidelines
- Has proper documentation

#### Reasoning Tasks
```yaml
task:
  type: "reasoning"
  prompt_template: |
    Think step-by-step to answer: {question}
  problems:
    - "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?"
```

Reward functions:
- Logical consistency
- Step-by-step breakdown
- Correct conclusion
- Clear reasoning

### Output

The fine-tuned model will be saved to:
- `./gpt-oss-grpo/final_model/`

Checkpoints are saved every 10 steps in:
- `./gpt-oss-grpo/checkpoint-*/`

### Understanding GRPO vs SFT

| Aspect | SFT (Supervised) | GRPO (Reinforcement) |
|--------|------------------|---------------------|
| **Training Data** | Explicit examples | Prompts + reward function |
| **Signal** | Imitate examples | Maximize reward |
| **Flexibility** | Fixed gold answers | Explores solutions |
| **Best For** | Known correct outputs | Complex reasoning, creativity |
| **Risk** | Limited to training data | Can game rewards (need safeguards) |

**When to use GRPO:**
- Math/reasoning where process matters
- Code generation with test-based rewards
- Alignment with human preferences
- Tasks with measurable outcomes

**When to use SFT:**
- Have high-quality example datasets
- Need predictable outputs
- Simpler tasks with clear correct answers

### Preventing Reward Hacking

GRPO models can sometimes "cheat" to maximize rewards. Safeguards in this example:

1. **Multiple reward components**: Balances different objectives
2. **Correctness checking**: Validates actual math accuracy
3. **Reasoning rewards**: Encourages explanation, not just answers
4. **Length penalties**: Prevents gaming with verbose responses

For production, consider:
- Test-based evaluation for code
- Human-in-the-loop validation
- Diverse test sets
- Adversarial prompts

### Monitoring Training

Watch the logs for reward metrics:
```
{'loss': 0.0002, 'grad_norm': 13.68, 'learning_rate': 0.0,
 'rewards/function_works/mean': -0.5, 'rewards/function_works/std': 2.12,
 'rewards/no_cheating/mean': -9.5, 'rewards/no_cheating/std': 14.85,
 'rewards/correctness_check/mean': 2.0, 'rewards/correctness_check/std': 2.83,
 'rewards/speed_check/mean': -2.06, 'rewards/speed_check/std': 2.92,
 'reward': -10.06, 'reward_std': 16.88, 'kl': 0.00179}
```

**Key Metrics:**
- **loss**: KL divergence penalty (β × KL), small values expected (0.0001-0.0002)
- **reward**: Total reward (sum of all reward functions), should increase over time
- **rewards/{function_name}/mean**: Individual reward component averages
- **kl**: KL divergence from reference policy, tracks policy drift
- **grad_norm**: Gradient magnitude, confirms backpropagation is working

**Good signs:**
- Total reward increases over time
- Individual rewards improve (e.g., correctness_check increases)
- KL stays relatively small (< 0.1) - policy not diverging too far
- grad_norm is non-zero and stable

**Warning signs:**
- Reward plateaus early (may need tuning)
- All reward from one component (likely gaming)
- Very high KL (> 1.0) - policy diverging too much from reference
- Gibberish with high rewards (reward hacking)

### Reference

This example is based on:
- [Unsloth GPT-OSS RL Documentation](https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning)
- [GRPO Colab Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb)
- [DeepSeek-R1 GRPO Paper](https://github.com/deepseek-ai/DeepSeek-R1) - Original GRPO method
