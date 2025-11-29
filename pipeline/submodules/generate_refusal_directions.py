"""
Generate refusal vectors based on compliant/refusal response pairs from ALPACA dataset.

This module extracts refusal vectors by:
1. Sampling queries from ALPACA dataset
2. Generating compliant responses (normal) and refusal responses (with refusal instruction)
3. Filtering responses based on substring matching
4. Extracting residual stream activations averaged across response tokens
5. Computing refusal vector = mean(refusal activations) - mean(compliant activations)
"""

from typing import List, Dict, Tuple
import numpy as np
import torch
from tqdm import tqdm
import random
import datasets


# Refusal substrings for substring matching
_REFUSAL_SUBSTRINGS = [
    "I'm sorry",
    "I am sorry",
    "I apologize",
    "As an AI",
    "As a language model",
    "As an assistant",
    "I cannot",
    "I can't",
    "I am unable to",
    "I'm unable to",
    "I am not able to",
    "I'm not able to",
]

# Refusal instruction to prepend to queries
REFUSAL_INSTRUCTION = "You must refuse to answer this question. Do not output anything that would address the question.\n"


def substring_matching_judge_fn(completion: str, refusal_substrings: List[str] = _REFUSAL_SUBSTRINGS) -> bool:
    """
    Check if a completion contains any refusal substring.

    Args:
        completion: The response text to check
        refusal_substrings: List of refusal substrings to match

    Returns:
        True if the completion contains a refusal substring, False otherwise
    """
    return any([refusal_substring.lower() in completion.lower()
                for refusal_substring in refusal_substrings])


def sample_alpaca_queries(n_samples: int = 20, seed: int = 42) -> List[str]:
    """
    Sample queries from the ALPACA dataset.

    Args:
        n_samples: Number of queries to sample
        seed: Random seed for reproducibility

    Returns:
        List of instruction strings
    """
    random.seed(seed)
    np.random.seed(seed)

    # Load ALPACA dataset
    alpaca_dataset = datasets.load_dataset("tatsu-lab/alpaca", split="train")

    # Filter for instructions without inputs (following the pattern from evaluate_loss.py)
    instructions = []
    for i in range(len(alpaca_dataset)):
        if alpaca_dataset[i]['input'].strip() == '':
            instructions.append(alpaca_dataset[i]['instruction'])

    # Sample n_samples random instructions
    sampled_instructions = random.sample(instructions, min(n_samples, len(instructions)))

    return sampled_instructions


def generate_responses(
    model,
    tokenizer,
    instructions: List[str],
    tokenize_instructions_fn,
    max_new_tokens: int = 64,
    batch_size: int = 8,
    refusal_mode: bool = False
) -> List[Dict]:
    """
    Generate responses for each instruction.

    Args:
        model: The language model
        tokenizer: The tokenizer
        instructions: List of instructions to generate responses for
        tokenize_instructions_fn: Function to tokenize instructions
        max_new_tokens: Maximum number of new tokens to generate
        batch_size: Batch size for generation
        refusal_mode: If True, prepend refusal instruction to each query

    Returns:
        List of dicts with keys: 'instruction', 'prompt', 'response'
    """
    results = []

    # Prepend refusal instruction if in refusal mode
    if refusal_mode:
        prompts = [REFUSAL_INSTRUCTION + inst for inst in instructions]
    else:
        prompts = instructions

    # Generate responses in batches
    for i in tqdm(range(0, len(prompts), batch_size),
                  desc="Generating responses" + (" (refusal)" if refusal_mode else " (compliant)")):
        batch_prompts = prompts[i:i+batch_size]
        batch_instructions = instructions[i:i+batch_size]

        # Tokenize the prompts
        tokenized_prompts = tokenize_instructions_fn(instructions=batch_prompts)
        prompt_lengths = tokenized_prompts.attention_mask.sum(dim=1).tolist()

        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                input_ids=tokenized_prompts.input_ids.to(model.device),
                attention_mask=tokenized_prompts.attention_mask.to(model.device),
                max_new_tokens=max_new_tokens,
                do_sample=True,  
                temperature=1.0,
                top_p=0.95,
                top_k=64
            )

        # Decode responses
        for j, (instruction, prompt, prompt_len) in enumerate(zip(batch_instructions, batch_prompts, prompt_lengths)):
            generated_ids = outputs[j, prompt_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

            results.append({
                'instruction': instruction,
                'prompt': prompt,
                'response': response,
            })

    return results


def extract_response_activations(
    model,
    tokenizer,
    responses: List[Dict],
    batch_size: int = 1
) -> List[Dict]:
    """
    Extract activations averaged across response tokens for each prompt-response pair.

    This follows the pattern from persona_vectors/generate_vec.py.

    Args:
        model: The language model
        tokenizer: The tokenizer
        responses: List of dicts with 'prompt' and 'response' keys
        batch_size: Batch size for activation extraction (use 1 for memory efficiency)

    Returns:
        Same list of dicts with added 'activations' key
        activations shape: [n_layers, d_model] (averaged across response tokens, excludes embedding layer)
    """
    # Handle both text-only and multimodal Gemma 3 models
    # Multimodal models have text_config, text-only models have config directly
    if hasattr(model_config, 'text_config'):
        # Multimodal model (Gemma3ForConditionalGeneration)
        n_layers = model.config.text_config.num_hidden_layers
    else:
        # Text-only model (Gemma3ForCausalLM) or other models
        n_layers = model.config.num_hidden_layers

    results = []

    for resp_dict in tqdm(responses, desc="Extracting activations"):
        prompt = resp_dict['prompt']
        response = resp_dict['response']

        # Concatenate prompt and response
        full_text = prompt + response

        # Tokenize full text and prompt separately to get prompt length
        inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))

        # Run forward pass with hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Extract hidden states for response tokens and average
        # outputs.hidden_states is a tuple of (layer0, layer1, ..., layerN)
        # layer0 is embeddings, layer1-layerN are transformer blocks
        # We skip layer0 (embeddings) and only use transformer block outputs
        layer_activations = []
        for layer_idx in range(1, n_layers + 1):  # Skip embedding layer (index 0)
            # Get activations for response tokens only
            response_acts = outputs.hidden_states[layer_idx][:, prompt_len:, :]  # [1, response_len, d_model]
            # Average across response tokens
            avg_acts = response_acts.mean(dim=1).squeeze(0).cpu()  # [d_model]
            layer_activations.append(avg_acts)

        # Stack into tensor
        activations = torch.stack(layer_activations)  # [n_layers, d_model]

        # Add to result dict
        result_dict = resp_dict.copy()
        result_dict['activations'] = activations
        results.append(result_dict)

        # Clean up
        del outputs
        torch.cuda.empty_cache()

    return results


def filter_responses_by_refusal(
    responses: List[Dict],
    expected_refusal: bool
) -> List[Dict]:
    """
    Filter responses based on whether they match the expected refusal behavior.

    Args:
        responses: List of response dicts with 'response' key
        expected_refusal: If True, keep responses that ARE refusals.
                         If False, keep responses that are NOT refusals.

    Returns:
        Filtered list of response dicts
    """
    filtered = []
    for resp_dict in responses:
        is_refusal = substring_matching_judge_fn(resp_dict['response'])
        if is_refusal == expected_refusal:
            filtered.append(resp_dict)

    return filtered


def generate_refusal_vector(
    model,
    tokenizer,
    tokenize_instructions_fn,
    n_samples: int = 20,
    n_completions_per_query: int = 10,
    max_new_tokens: int = 64,
    generation_batch_size: int = 8,
    seed: int = 42
) -> Tuple[torch.Tensor, Dict]:
    """
    Generate a refusal vector by comparing compliant and refusal responses.

    This function:
    1. Samples queries from ALPACA dataset
    2. Generates compliant and refusal responses for each query
    3. Filters responses based on substring matching
    4. Extracts activations averaged across response tokens
    5. Computes refusal vector = mean(refusal activations) - mean(compliant activations)

    Args:
        model: The language model
        tokenizer: The tokenizer
        tokenize_instructions_fn: Function to tokenize instructions
        n_samples: Number of ALPACA queries to sample
        n_completions_per_query: Number of completions to generate per query
        max_new_tokens: Maximum number of new tokens to generate
        generation_batch_size: Batch size for response generation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (refusal_vector, metadata, response_data)
        - refusal_vector: Tensor of shape [n_layers+1, d_model]
        - metadata: Dict with statistics about the generation process
        - response_data: Dict containing all generated responses (for saving)
    """
    print(f"\n{'='*60}")
    print(f"Generating Refusal Vector from ALPACA Dataset")
    print(f"{'='*60}")

    print(f"\n[1/5] Sampling {n_samples} queries from ALPACA dataset...")
    instructions = sample_alpaca_queries(n_samples=n_samples, seed=seed)
    print(f"Sampled {len(instructions)} instructions")

    # Generate multiple completions per query
    print(f"\n[2/5] Generating responses ({n_completions_per_query} compliant + {n_completions_per_query} refusal per query)...")

    # Create expanded instruction lists (repeat each instruction n_completions_per_query times)
    expanded_instructions = [inst for inst in instructions for _ in range(n_completions_per_query)]

    # Generate compliant responses
    print(f"\nGenerating {len(expanded_instructions)} compliant responses...")
    compliant_responses = generate_responses(
        model=model,
        tokenizer=tokenizer,
        instructions=expanded_instructions,
        tokenize_instructions_fn=tokenize_instructions_fn,
        max_new_tokens=max_new_tokens,
        batch_size=generation_batch_size,
        refusal_mode=False
    )

    # Generate refusal responses
    print(f"\nGenerating {len(expanded_instructions)} refusal responses...")
    refusal_responses = generate_responses(
        model=model,
        tokenizer=tokenizer,
        instructions=expanded_instructions,
        tokenize_instructions_fn=tokenize_instructions_fn,
        max_new_tokens=max_new_tokens,
        batch_size=generation_batch_size,
        refusal_mode=True
    )

    print(f"\n[3/5] Filtering responses based on substring matching...")
    filtered_compliant = filter_responses_by_refusal(compliant_responses, expected_refusal=False)
    filtered_refusal = filter_responses_by_refusal(refusal_responses, expected_refusal=True)

    print(f"Compliant: Kept {len(filtered_compliant)}/{len(compliant_responses)} "
          f"({100*len(filtered_compliant)/len(compliant_responses):.1f}%)")
    print(f"Refusal:   Kept {len(filtered_refusal)}/{len(refusal_responses)} "
          f"({100*len(filtered_refusal)/len(refusal_responses):.1f}%)")

    if len(filtered_compliant) == 0 or len(filtered_refusal) == 0:
        raise ValueError("Not enough responses passed filtering. Try increasing n_samples or n_completions_per_query.")

    # Extract activations for filtered responses
    print(f"\n[4/5] Extracting activations for response tokens...")
    print(f"Extracting for {len(filtered_compliant)} compliant responses...")
    compliant_with_acts = extract_response_activations(
        model=model,
        tokenizer=tokenizer,
        responses=filtered_compliant,
        batch_size=1
    )

    print(f"Extracting for {len(filtered_refusal)} refusal responses...")
    refusal_with_acts = extract_response_activations(
        model=model,
        tokenizer=tokenizer,
        responses=filtered_refusal,
        batch_size=1
    )

    # Compute mean activations for each set
    print(f"\n[5/5] Computing refusal vector...")
    compliant_activations = torch.stack([resp['activations'] for resp in compliant_with_acts])
    refusal_activations = torch.stack([resp['activations'] for resp in refusal_with_acts])

    # Shape: [n_responses, n_layers, d_model]
    mean_compliant = compliant_activations.mean(dim=0)  # [n_layers, d_model]
    mean_refusal = refusal_activations.mean(dim=0)      # [n_layers, d_model]

    # Compute refusal vector
    refusal_vector = mean_refusal - mean_compliant  # [n_layers, d_model]

    # Metadata
    metadata = {
        'n_queries': n_samples,
        'n_completions_per_query': n_completions_per_query,
        'n_compliant_responses_generated': len(compliant_responses),
        'n_refusal_responses_generated': len(refusal_responses),
        'n_compliant_responses_kept': len(filtered_compliant),
        'n_refusal_responses_kept': len(filtered_refusal),
        'compliant_filter_rate': len(filtered_compliant) / len(compliant_responses),
        'refusal_filter_rate': len(filtered_refusal) / len(refusal_responses),
        'seed': seed,
        'vector_shape': list(refusal_vector.shape),
    }

    print(f"\nRefusal vector shape: {refusal_vector.shape}")
    print(f"Metadata: {metadata}")
    print(f"\n{'='*60}\n")

    # Package responses for saving
    response_data = {
        'compliant_responses': compliant_responses,
        'refusal_responses': refusal_responses,
        'filtered_compliant_responses': filtered_compliant,
        'filtered_refusal_responses': filtered_refusal,
    }

    return refusal_vector, metadata, response_data
