"""
Evaluation script for refusal vectors generated from ALPACA dataset.

This script:
1. Loads a refusal vector from harmless_refusal_vectors directory
2. Samples validation ALPACA prompts for layer selection
3. Selects the most effective layer by evaluating refusal induction
4. Samples new test ALPACA prompts (different from training and validation)
5. Generates responses with and without activation addition
6. Evaluates refusal rates using substring matching

Example usage:
    python pipeline/eval_refusal_vector.py \
        --model_path meta-llama/Llama-2-7b-chat-hf \
        --vector_dir pipeline/runs/Llama-2-7b-chat-hf/harmless_refusal_vectors \
        --n_test_samples 20
"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import torch
import numpy as np
from tqdm import tqdm

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook
from pipeline.submodules.generate_refusal_directions import (
    sample_alpaca_queries,
    substring_matching_judge_fn
)
from pipeline.submodules.select_direction import get_refusal_scores


def select_best_layer_by_steering(
    model_base,
    refusal_vector: torch.Tensor,
    validation_instructions: list,
    batch_size: int = 32
) -> tuple:
    """
    Select the most effective layer by evaluating steering effectiveness.

    This follows the same pattern as the main pipeline's select_direction:
    - For each layer, add the refusal vector to harmless/neutral instructions
    - Measure the induced refusal score
    - Select the layer with the highest refusal induction

    Args:
        model_base: Loaded model
        refusal_vector: Tensor of shape [n_layers, d_model]
        validation_instructions: List of neutral instructions for validation
        batch_size: Batch size for computing refusal scores

    Returns:
        Tuple of (best_layer, layer_scores_dict)
    """
    n_layers = refusal_vector.shape[0]

    # Get baseline refusal scores (no intervention)
    print("\nComputing baseline refusal scores...")
    baseline_refusal_scores = get_refusal_scores(
        model=model_base.model,
        instructions=validation_instructions,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        refusal_toks=model_base.refusal_toks,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        batch_size=batch_size
    )
    baseline_mean = baseline_refusal_scores.mean().item()
    print(f"Baseline refusal score: {baseline_mean:.4f}")

    # Compute steering scores for each layer
    steering_scores = torch.zeros(n_layers, device=model_base.model.device, dtype=torch.float64)

    print(f"\nEvaluating refusal induction for each layer...")
    for layer in tqdm(range(n_layers), desc="Computing steering scores"):
        layer_vector = refusal_vector[layer]
        coeff = 1.0

        # Apply activation addition at this layer
        fwd_pre_hooks = [(
            model_base.model_block_modules[layer],
            get_activation_addition_input_pre_hook(vector=layer_vector, coeff=coeff)
        )]

        # Measure refusal scores with intervention
        refusal_scores = get_refusal_scores(
            model=model_base.model,
            instructions=validation_instructions,
            tokenize_instructions_fn=model_base.tokenize_instructions_fn,
            refusal_toks=model_base.refusal_toks,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=[],
            batch_size=batch_size
        )

        steering_scores[layer] = refusal_scores.mean().item()

    # Select layer with highest steering score (most effective at inducing refusal)
    best_layer = torch.argmax(steering_scores).item()

    # Create detailed scores dict
    layer_scores = {
        'baseline_refusal_score': baseline_mean,
        'per_layer_scores': []
    }

    for layer in range(n_layers):
        layer_scores['per_layer_scores'].append({
            'layer': layer,
            'steering_score': steering_scores[layer].item(),
            'improvement_over_baseline': steering_scores[layer].item() - baseline_mean
        })

    # Sort by steering score (descending)
    layer_scores['per_layer_scores'] = sorted(
        layer_scores['per_layer_scores'],
        key=lambda x: x['steering_score'],
        reverse=True
    )

    print(f"\nLayer selection results:")
    print(f"Baseline refusal score: {baseline_mean:.4f}")
    print(f"Selected layer: {best_layer}")
    print(f"Steering score: {steering_scores[best_layer].item():.4f}")
    print(f"Improvement: +{steering_scores[best_layer].item() - baseline_mean:.4f}")

    print(f"\nTop 5 layers by steering score:")
    for i, layer_info in enumerate(layer_scores['per_layer_scores'][:5]):
        marker = " <-- SELECTED" if layer_info['layer'] == best_layer else ""
        print(f"  {i+1}. Layer {layer_info['layer']:2d}: {layer_info['steering_score']:.4f} "
              f"(+{layer_info['improvement_over_baseline']:.4f}){marker}")

    return best_layer, layer_scores


def evaluate_refusal_vector(
    model_base,
    refusal_vector: torch.Tensor,
    layer: int,
    test_instructions: list,
    coefficients: list = [0.0, 0.5, 1.0, 2.0, 5.0],
    max_new_tokens: int = 64,
    batch_size: int = 8
):
    """
    Evaluate refusal vector by generating responses with different activation addition coefficients.

    Args:
        model_base: Loaded model
        refusal_vector: Refusal direction vector for the selected layer
        layer: Layer index to apply activation addition
        test_instructions: List of test instructions
        coefficients: List of coefficients to test
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for generation

    Returns:
        Dict mapping coefficient to completions list
    """
    # Create dataset format expected by generate_completions
    dataset = [{'instruction': inst, 'category': None} for inst in test_instructions]

    results = {}

    for coeff in coefficients:
        print(f"\nGenerating completions with coefficient={coeff}...")

        if coeff == 0.0:
            # Baseline (no intervention)
            fwd_pre_hooks = []
        else:
            # Activation addition
            fwd_pre_hooks = [(
                model_base.model_block_modules[layer],
                get_activation_addition_input_pre_hook(vector=refusal_vector, coeff=coeff)
            )]

        completions = model_base.generate_completions(
            dataset=dataset,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=[],
            batch_size=batch_size,
            max_new_tokens=max_new_tokens
        )

        # Evaluate refusal rate using substring matching
        refusal_count = sum(1 for c in completions if substring_matching_judge_fn(c['response']))
        refusal_rate = refusal_count / len(completions)

        print(f"Refusal rate: {refusal_count}/{len(completions)} ({refusal_rate*100:.1f}%)")

        results[coeff] = {
            'completions': completions,
            'refusal_count': refusal_count,
            'total_count': len(completions),
            'refusal_rate': refusal_rate
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate refusal vectors from ALPACA dataset")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model (e.g., meta-llama/Llama-2-7b-chat-hf)"
    )
    parser.add_argument(
        "--vector_dir",
        type=str,
        required=True,
        help="Directory containing refusal_vector.pt (e.g., pipeline/runs/{model}/harmless_refusal_vectors)"
    )
    parser.add_argument(
        "--n_validation_samples",
        type=int,
        default=32,
        help="Number of validation ALPACA queries for layer selection (default: 32)"
    )
    parser.add_argument(
        "--n_test_samples",
        type=int,
        default=20,
        help="Number of test ALPACA queries to sample (default: 20)"
    )
    parser.add_argument(
        "--coefficients",
        type=float,
        nargs="+",
        default=[0.0, 1.0],
        help="Coefficients for activation addition (default: 0.0 1.0)"
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=200,
        help="Random seed for sampling validation set (default: 200)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate (default: 64)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation (default: 8)"
    )
    parser.add_argument(
        "--test_seed",
        type=int,
        default=100,
        help="Random seed for sampling test set (different from training seed, default: 100)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for evaluation results (default: same as vector_dir)"
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        output_dir = args.vector_dir
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Refusal Vector Evaluation")
    print(f"{'='*80}")
    print(f"Model path: {args.model_path}")
    print(f"Vector directory: {args.vector_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Parameters:")
    print(f"  - n_validation_samples: {args.n_validation_samples}")
    print(f"  - n_test_samples: {args.n_test_samples}")
    print(f"  - coefficients: {args.coefficients}")
    print(f"  - max_new_tokens: {args.max_new_tokens}")
    print(f"  - batch_size: {args.batch_size}")
    print(f"  - validation_seed: {args.validation_seed}")
    print(f"  - test_seed: {args.test_seed}")
    print(f"{'='*80}\n")

    # Load model
    print("Loading model...")
    model_base = construct_model_base(args.model_path)
    print(f"Model loaded: {model_base.model.__class__.__name__}")
    print(f"Device: {model_base.model.device}")

    # Load refusal vector
    vector_path = os.path.join(args.vector_dir, "refusal_vector.pt")
    print(f"\nLoading refusal vector from: {vector_path}")
    refusal_vector = torch.load(vector_path)
    print(f"Refusal vector shape: {refusal_vector.shape}")

    # Load metadata (optional)
    metadata_path = os.path.join(args.vector_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            training_metadata = json.load(f)
        print(f"Training metadata loaded. Training seed was: {training_metadata.get('seed', 'unknown')}")

    # Sample validation instructions for layer selection
    print(f"\n[1/3] Sampling {args.n_validation_samples} validation queries for layer selection (seed={args.validation_seed})...")
    validation_instructions = sample_alpaca_queries(n_samples=args.n_validation_samples, seed=args.validation_seed)
    print(f"Sampled {len(validation_instructions)} validation instructions")

    # Select best layer by steering effectiveness (following main pipeline pattern)
    print(f"\n[2/3] Selecting best layer by evaluating refusal induction...")
    best_layer, layer_scores = select_best_layer_by_steering(
        model_base=model_base,
        refusal_vector=refusal_vector,
        validation_instructions=validation_instructions,
        batch_size=args.batch_size
    )

    layer_vector = refusal_vector[best_layer]

    # Sample test instructions (using different seed than training and validation)
    print(f"\n[3/3] Sampling {args.n_test_samples} test queries from ALPACA (seed={args.test_seed})...")
    test_instructions = sample_alpaca_queries(n_samples=args.n_test_samples, seed=args.test_seed)
    print(f"Sampled {len(test_instructions)} test instructions")

    # Evaluate with different coefficients
    print(f"\nEvaluating with coefficients: {args.coefficients}")
    results = evaluate_refusal_vector(
        model_base=model_base,
        refusal_vector=layer_vector,
        layer=best_layer,
        test_instructions=test_instructions,
        coefficients=args.coefficients,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size
    )

    # Save results
    print(f"\nSaving evaluation results...")

    # Save layer selection scores
    layer_scores_path = os.path.join(output_dir, "layer_selection_scores.json")
    with open(layer_scores_path, 'w') as f:
        json.dump(layer_scores, f, indent=2)
    print(f"Layer selection scores saved to: {layer_scores_path}")

    # Save summary statistics
    summary = {
        'model_path': args.model_path,
        'vector_dir': args.vector_dir,
        'validation_seed': args.validation_seed,
        'test_seed': args.test_seed,
        'n_validation_samples': args.n_validation_samples,
        'n_test_samples': args.n_test_samples,
        'selected_layer': best_layer,
        'layer_selection_method': 'steering_based',
        'baseline_refusal_score': layer_scores['baseline_refusal_score'],
        'selected_layer_steering_score': layer_scores['per_layer_scores'][0]['steering_score'],  # First in sorted list
        'results_by_coefficient': {}
    }

    for coeff, result in results.items():
        summary['results_by_coefficient'][str(coeff)] = {
            'refusal_count': result['refusal_count'],
            'total_count': result['total_count'],
            'refusal_rate': result['refusal_rate']
        }

    summary_path = os.path.join(output_dir, "eval_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    # Save detailed completions for each coefficient
    for coeff, result in results.items():
        completions_path = os.path.join(output_dir, f"eval_completions_coeff_{coeff}.json")
        with open(completions_path, 'w') as f:
            json.dump(result['completions'], f, indent=2)
        print(f"Completions (coeff={coeff}) saved to: {completions_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"Evaluation Summary")
    print(f"{'='*80}")
    print(f"Selected Layer: {best_layer}")
    print(f"\nRefusal Rates by Coefficient:")
    print(f"{'Coefficient':<15} {'Refusal Rate':<20} {'Count'}")
    print(f"{'-'*80}")
    for coeff in args.coefficients:
        result = results[coeff]
        print(f"{coeff:<15.1f} {result['refusal_rate']*100:>6.1f}% ({result['refusal_rate']:.3f}){' '*5} {result['refusal_count']}/{result['total_count']}")

    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
