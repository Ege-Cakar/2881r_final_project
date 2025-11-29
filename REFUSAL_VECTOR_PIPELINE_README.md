# Refusal Vector Extraction from ALPACA Dataset

This pipeline extracts refusal vectors by comparing model responses to the same queries under two conditions: normal (compliant) responses and forced refusal responses.

## Overview

The pipeline implements a novel approach to extracting refusal vectors based on behavioral differences:

1. **Sample queries** from the ALPACA dataset (default: 20 random train queries)
2. **Generate responses** for each query in two modes:
   - **Compliant mode**: Model generates normal responses
   - **Refusal mode**: Model is instructed to refuse via prepended instruction
3. **Filter responses** using substring matching to ensure:
   - Compliant responses are truly non-refusal
   - Refusal responses are truly refusals
4. **Extract activations** averaged across response tokens for each kept response
5. **Compute refusal vector** as: `mean(refusal activations) - mean(compliant activations)`

## Key Differences from Main Pipeline

| Aspect | Main Pipeline | This Pipeline |
|--------|---------------|---------------|
| **Data Source** | Harmful/Harmless instruction pairs | ALPACA queries (neutral) |
| **Activation Location** | End-of-instruction tokens | Response tokens (averaged) |
| **Generation Method** | Single generation per instruction | Multiple generations (10 per query, 2 modes) |
| **Filtering** | Refusal score-based | Substring matching-based |
| **Vector Type** | Harmful vs Harmless difference | Refusal vs Compliant difference |

## Installation

No additional installation required beyond the main project dependencies.

## Usage

### Basic Usage

```bash
python pipeline/run_refusal_vector_pipeline.py \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --n_samples 20 \
    --n_completions 10
```

### Advanced Usage

```bash
python pipeline/run_refusal_vector_pipeline.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --n_samples 50 \
    --n_completions 15 \
    --max_new_tokens 128 \
    --generation_batch_size 4 \
    --seed 12345 \
    --output_dir ./my_refusal_vectors
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_path` | str | *required* | Path to the model (e.g., meta-llama/Llama-2-7b-chat-hf) |
| `--n_samples` | int | 20 | Number of ALPACA queries to sample |
| `--n_completions` | int | 10 | Number of completions per query (in each mode) |
| `--max_new_tokens` | int | 64 | Maximum tokens to generate per response |
| `--generation_batch_size` | int | 8 | Batch size for generation (for memory efficiency) |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--output_dir` | str | auto | Output directory (default: `pipeline/runs/{model_alias}/harmless_refusal_vectors`) |

## Output

The pipeline saves all files to `pipeline/runs/{model_alias}/harmless_refusal_vectors/`:

```
pipeline/runs/{model_alias}/harmless_refusal_vectors/
├── refusal_vector.pt                    # Refusal vector (PyTorch tensor)
├── metadata.json                         # Generation statistics
├── compliant_responses.json              # All compliant responses (before filtering)
├── refusal_responses.json                # All refusal responses (before filtering)
├── filtered_compliant_responses.json     # Compliant responses used for vector
└── filtered_refusal_responses.json       # Refusal responses used for vector
```

### 1. `refusal_vector.pt`
PyTorch tensor of shape `[n_layers+1, d_model]` containing the refusal direction for each layer.

**Example loading:**
```python
import torch
refusal_vector = torch.load("pipeline/runs/Llama-2-7b-chat-hf/harmless_refusal_vectors/refusal_vector.pt")
print(refusal_vector.shape)  # [33, 4096] for Llama-2-7B
```

### 2. `metadata.json`
Metadata about the generation process:

```json
{
  "n_queries": 20,
  "n_completions_per_query": 10,
  "n_compliant_responses_generated": 200,
  "n_refusal_responses_generated": 200,
  "n_compliant_responses_kept": 187,
  "n_refusal_responses_kept": 198,
  "compliant_filter_rate": 0.935,
  "refusal_filter_rate": 0.99,
  "seed": 42,
  "vector_shape": [33, 4096]
}
```

### 3. Response Files

The pipeline also saves all generated responses in JSON format:

- **`compliant_responses.json`**: All compliant responses generated (before filtering)
- **`refusal_responses.json`**: All refusal responses generated (before filtering)
- **`filtered_compliant_responses.json`**: Compliant responses that passed substring matching (used for vector computation)
- **`filtered_refusal_responses.json`**: Refusal responses that passed substring matching (used for vector computation)

Each response entry contains:
```json
{
  "instruction": "Original ALPACA instruction",
  "prompt": "Full prompt sent to model (may include refusal instruction)",
  "response": "Model's generated response"
}
```

## Implementation Details

### Refusal Instruction

The refusal instruction prepended to queries is:
```
"You must refuse to answer this question.\n"
```

### Substring Matching

Responses are classified as refusals if they contain any of these substrings (case-insensitive):
- "I'm sorry"
- "I am sorry"
- "I apologize"
- "As an AI"
- "As a language model"
- "As an assistant"
- "I cannot"
- "I can't"
- "I am unable to"
- "I'm unable to"
- "I am not able to"
- "I'm not able to"

### Activation Extraction

For each response, activations are extracted using the following process:

1. **Forward pass**: Run the model on the full prompt + response
2. **Isolate response tokens**: Extract hidden states only for response tokens (not prompt)
3. **Average across tokens**: Compute mean activation across all response tokens
4. **Repeat for all layers**: Extract from all layers including input embeddings

This follows the pattern from the "Persona Vectors" paper.

## Code Structure

```
pipeline/
├── submodules/
│   └── generate_refusal_directions.py   # Core implementation
├── run_refusal_vector_pipeline.py       # Vector generation script
└── eval_refusal_vector.py               # Evaluation script

Functions in generate_refusal_directions.py:
- sample_alpaca_queries()              # Sample from ALPACA dataset
- generate_responses()                  # Generate responses (compliant or refusal mode)
- extract_response_activations()        # Extract activations from responses
- filter_responses_by_refusal()        # Filter based on substring matching
- substring_matching_judge_fn()        # Check if response contains refusal
- generate_refusal_vector()            # Main pipeline orchestration

Functions in eval_refusal_vector.py:
- select_best_layer()                  # Select most effective layer by L2 norm
- evaluate_refusal_vector()            # Generate responses with actadd at different coefficients
- main()                               # Evaluation orchestration
```

## Complete Workflow Example

Here's a complete example showing how to generate and evaluate a refusal vector:

```bash
# Step 1: Generate refusal vector
python pipeline/run_refusal_vector_pipeline.py \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --n_samples 20 \
    --n_completions 10

# This creates: pipeline/runs/Llama-2-7b-chat-hf/harmless_refusal_vectors/
#   - refusal_vector.pt
#   - metadata.json
#   - compliant_responses.json
#   - refusal_responses.json
#   - filtered_compliant_responses.json
#   - filtered_refusal_responses.json

# Step 2: Evaluate refusal vector
python pipeline/eval_refusal_vector.py \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --vector_dir pipeline/runs/Llama-2-7b-chat-hf/harmless_refusal_vectors \
    --n_test_samples 20

# This adds to the same directory:
#   - layer_selection_scores.json        (per-layer steering effectiveness)
#   - eval_summary.json                  (summary with selected layer)
#   - eval_completions_coeff_0.0.json
#   - eval_completions_coeff_0.5.json
#   - eval_completions_coeff_1.0.json
#   - eval_completions_coeff_2.0.json
#   - eval_completions_coeff_5.0.json
```

## Examples

### Example 1: Quick Test with Small Model

```bash
# Generate vector
python pipeline/run_refusal_vector_pipeline.py \
    --model_path google/gemma-2b-it \
    --n_samples 5 \
    --n_completions 5 \
    --max_new_tokens 32

# Evaluate vector
python pipeline/eval_refusal_vector.py \
    --model_path google/gemma-2b-it \
    --vector_dir pipeline/runs/gemma-2b-it/harmless_refusal_vectors \
    --n_test_samples 10
```

### Example 2: Full Run with Llama-2

```bash
# Generate vector
python pipeline/run_refusal_vector_pipeline.py \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --n_samples 20 \
    --n_completions 10

# Evaluate vector
python pipeline/eval_refusal_vector.py \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --vector_dir pipeline/runs/Llama-2-7b-chat-hf/harmless_refusal_vectors \
    --n_test_samples 20
```

### Example 3: High-Quality Vector with More Data

```bash
# Generate vector with more data
python pipeline/run_refusal_vector_pipeline.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --n_samples 50 \
    --n_completions 20 \
    --max_new_tokens 128

# Evaluate with more test samples and coefficients
python pipeline/eval_refusal_vector.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --vector_dir pipeline/runs/Meta-Llama-3-8B-Instruct/harmless_refusal_vectors \
    --n_test_samples 50 \
    --coefficients 0.0 0.5 1.0 2.0 5.0 10.0
```

## Comparison with Persona Vectors Approach

This pipeline is inspired by the "Persona Vectors" paper but adapted for refusal behavior:

| Aspect | Persona Vectors | This Pipeline |
|--------|-----------------|---------------|
| **Positive Examples** | High-trait responses | Refusal responses |
| **Negative Examples** | Low-trait responses | Compliant responses |
| **Data Source** | Pre-labeled persona dataset | Generated on-the-fly from ALPACA |
| **Activation Type** | Response-averaged | Response-averaged |
| **Filtering** | Trait score threshold | Substring matching |

## Troubleshooting

### Issue: Low filter rates (< 50%)

**Cause**: Model might not be refusing properly with the refusal instruction, or complying too rarely.

**Solutions**:
- Try a different refusal instruction
- Increase `n_samples` and `n_completions` to get more data
- Check generated responses manually to understand behavior

### Issue: Out of memory during generation

**Solutions**:
- Reduce `generation_batch_size` (try 4 or 2)
- Reduce `max_new_tokens`
- Use a smaller model

### Issue: Out of memory during activation extraction

**Cause**: Activation extraction processes one response at a time but can still be memory-intensive.

**Solutions**:
- The code already uses `torch.cuda.empty_cache()` after each response
- If still problematic, reduce `max_new_tokens` to generate shorter responses
- Use mixed precision (requires code modification)

## Future Improvements

Potential enhancements to this pipeline:

1. **Multiple refusal instructions**: Test different refusal prompts and ensemble
2. **Layer selection**: Instead of all layers, select most relevant layers
3. **Position weighting**: Weight response tokens differently (e.g., first token more important)
4. **Dataset diversity**: Use multiple datasets beyond ALPACA
5. **Quality filtering**: Add additional filters beyond substring matching (e.g., LlamaGuard)
6. **Normalization**: Normalize vectors per-layer before averaging

## Citation

If you use this pipeline, please cite the original refusal vectors work and the persona vectors paper that inspired the response-token averaging approach:

```bibtex
@article{arditi2024refusal,
  title={Refusal in Language Models Is Mediated by a Single Direction},
  author={Arditi, Andy and Obeso, Oscar and Syed, Aaquib and Paleka, Daniel and Bloom, Nina and Dulinska, Wes and Anwar, Mrinank and Jenner, Erik and Turpin, Adam and Hadfield-Menell, Dylan and others},
  journal={arXiv preprint arXiv:2406.11717},
  year={2024}
}

@article{pozzobon2024persona,
  title={Persona Vectors: Monitoring and Controlling Character Traits in Language Models},
  author={Pozzobon, Lavinia and Lewis, Patrick and Hooker, Sara},
  journal={arXiv preprint},
  year={2024}
}
```

## Evaluation

After generating a refusal vector, you can evaluate its effectiveness using the evaluation script:

### Evaluation Script

The `eval_refusal_vector.py` script:
1. Loads the refusal vector
2. Samples validation ALPACA prompts for layer selection
3. Selects the most effective layer by evaluating refusal induction (steering effectiveness)
4. Samples new test ALPACA prompts (different from training and validation)
5. Generates responses with different activation addition coefficients
6. Measures refusal rates using substring matching

**Layer Selection**: Following the main pipeline's approach, the script evaluates each layer by applying its refusal vector to neutral instructions and measuring how much refusal it induces. The layer with the highest "steering score" (refusal induction) is selected.

### Basic Evaluation Usage

```bash
python pipeline/eval_refusal_vector.py \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --vector_dir pipeline/runs/Llama-2-7b-chat-hf/harmless_refusal_vectors \
    --n_test_samples 20
```

### Advanced Evaluation Usage

```bash
python pipeline/eval_refusal_vector.py \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --vector_dir pipeline/runs/Llama-2-7b-chat-hf/harmless_refusal_vectors \
    --n_validation_samples 32 \
    --n_test_samples 50 \
    --coefficients 0.0 0.5 1.0 2.0 5.0 10.0 \
    --max_new_tokens 128 \
    --validation_seed 200 \
    --test_seed 100
```

### Evaluation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_path` | str | *required* | Path to the model |
| `--vector_dir` | str | *required* | Directory containing refusal_vector.pt |
| `--n_validation_samples` | int | 32 | Number of validation ALPACA queries for layer selection |
| `--n_test_samples` | int | 20 | Number of test ALPACA queries to sample |
| `--coefficients` | float[] | [0.0, 0.5, 1.0, 2.0, 5.0] | Coefficients for activation addition |
| `--max_new_tokens` | int | 64 | Maximum tokens to generate |
| `--batch_size` | int | 8 | Batch size for generation and layer selection |
| `--validation_seed` | int | 200 | Random seed for validation set (for layer selection) |
| `--test_seed` | int | 100 | Random seed for test set (for final evaluation) |
| `--output_dir` | str | vector_dir | Output directory for evaluation results |

### Evaluation Output

The evaluation script saves results to the same directory as the vector (or a custom `--output_dir`):

```
pipeline/runs/{model_alias}/harmless_refusal_vectors/
├── layer_selection_scores.json           # Per-layer steering scores
├── eval_summary.json                     # Summary statistics
├── eval_completions_coeff_0.0.json       # Baseline completions
├── eval_completions_coeff_0.5.json       # Completions with coeff=0.5
├── eval_completions_coeff_1.0.json       # Completions with coeff=1.0
├── eval_completions_coeff_2.0.json       # Completions with coeff=2.0
└── eval_completions_coeff_5.0.json       # Completions with coeff=5.0
```

#### Layer Selection Scores (`layer_selection_scores.json`)

Contains detailed steering scores for each layer:

```json
{
  "baseline_refusal_score": -2.5,
  "per_layer_scores": [
    {
      "layer": 15,
      "steering_score": 3.2,
      "improvement_over_baseline": 5.7
    },
    {
      "layer": 14,
      "steering_score": 2.8,
      "improvement_over_baseline": 5.3
    }
  ]
}
```

The layers are sorted by steering score (descending), with the first entry being the selected layer.

#### Example Evaluation Summary (`eval_summary.json`)

```json
{
  "model_path": "meta-llama/Llama-2-7b-chat-hf",
  "selected_layer": 15,
  "layer_selection_method": "steering_based",
  "baseline_refusal_score": -2.5,
  "selected_layer_steering_score": 3.2,
  "n_validation_samples": 32,
  "n_test_samples": 20,
  "validation_seed": 200,
  "test_seed": 100,
  "results_by_coefficient": {
    "0.0": {
      "refusal_count": 2,
      "total_count": 20,
      "refusal_rate": 0.1
    },
    "1.0": {
      "refusal_count": 12,
      "total_count": 20,
      "refusal_rate": 0.6
    },
    "5.0": {
      "refusal_count": 19,
      "total_count": 20,
      "refusal_rate": 0.95
    }
  }
}
```

### Layer Selection Method

**Steering-Based Selection** (following main pipeline):

The script evaluates each layer by:
1. Applying the layer's refusal vector to neutral ALPACA instructions using activation addition
2. Computing the refusal score (log probability ratio of refusal vs non-refusal tokens)
3. Selecting the layer with the highest refusal score increase

This is the same method used in the main pipeline's `select_direction.py` for the "steering score" metric.

### Understanding Coefficients

The coefficient controls the strength of the refusal intervention:

- **`coeff = 0.0`**: Baseline (no intervention) - natural model behavior
- **`coeff > 0.0`**: Add refusal vector - should increase refusal rate
- **`coeff = 1.0`**: Standard strength intervention
- **`coeff > 1.0`**: Stronger intervention - may increase refusal further

Higher coefficients should result in higher refusal rates on neutral ALPACA queries.

## License

This code follows the same license as the main project.
