"""
Runner script for generating refusal vectors from ALPACA dataset.

This script generates refusal vectors by comparing compliant and refusal responses
to randomly sampled ALPACA queries.

Example usage:
    python pipeline/run_refusal_vector_pipeline.py --model_path meta-llama/Llama-2-7b-chat-hf --n_samples 20 --n_completions 10
"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import torch
from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_refusal_directions import generate_refusal_vector


def main():
    parser = argparse.ArgumentParser(description="Generate refusal vectors from ALPACA dataset")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model (e.g., meta-llama/Llama-2-7b-chat-hf)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=20,
        help="Number of ALPACA queries to sample (default: 20)"
    )
    parser.add_argument(
        "--n_completions",
        type=int,
        default=10,
        help="Number of completions per query (default: 10)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate (default: 64)"
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=8,
        help="Batch size for response generation (default: 8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for saving refusal vectors (default: pipeline/runs/{model_alias}/harmless_refusal_vectors)"
    )

    args = parser.parse_args()

    # Create model alias from model path (following main pipeline pattern)
    model_alias = os.path.basename(args.model_path)
    config = Config(model_alias=model_alias, model_path=args.model_path)

    # Set output directory
    if args.output_dir is None:
        output_dir = os.path.join(config.artifact_path(), "harmless_refusal_vectors")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Refusal Vector Generation Pipeline")
    print(f"{'='*80}")
    print(f"Model alias: {model_alias}")
    print(f"Model path: {config.model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Parameters:")
    print(f"  - n_samples: {args.n_samples}")
    print(f"  - n_completions_per_query: {args.n_completions}")
    print(f"  - max_new_tokens: {args.max_new_tokens}")
    print(f"  - generation_batch_size: {args.generation_batch_size}")
    print(f"  - seed: {args.seed}")
    print(f"{'='*80}\n")

    # Load model
    print("Loading model...")
    model_base = construct_model_base(config.model_path)
    print(f"Model loaded: {model_base.model.__class__.__name__}")
    print(f"Device: {model_base.model.device}")

    # Handle both text-only and multimodal Gemma 3 models
    # Multimodal models have text_config, text-only models have config directly
    model_config = model_base.model.config
    if hasattr(model_config, 'text_config'):
        # Multimodal model (Gemma3ForConditionalGeneration)
        text_config = model_config.text_config
        print(f"Number of layers: {text_config.num_hidden_layers}")
        print(f"Hidden size: {text_config.hidden_size}")
    else:
        # Text-only model (Gemma3ForCausalLM) or other models
        print(f"Number of layers: {model_config.num_hidden_layers}")
        print(f"Hidden size: {model_config.hidden_size}")

    # Generate refusal vector
    refusal_vector, metadata, response_data = generate_refusal_vector(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        n_samples=args.n_samples,
        n_completions_per_query=args.n_completions,
        max_new_tokens=args.max_new_tokens,
        generation_batch_size=args.generation_batch_size,
        seed=args.seed
    )

    # Save refusal vector
    vector_path = os.path.join(output_dir, "refusal_vector.pt")
    torch.save(refusal_vector, vector_path)
    print(f"\nRefusal vector saved to: {vector_path}")

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    # Save generated responses (remove activations for JSON serialization)
    def remove_activations(responses):
        """Remove activation tensors from response dicts for JSON serialization."""
        cleaned = []
        for resp in responses:
            cleaned_resp = {k: v for k, v in resp.items() if k != 'activations'}
            cleaned.append(cleaned_resp)
        return cleaned

    # Save all compliant responses
    compliant_path = os.path.join(output_dir, "compliant_responses.json")
    with open(compliant_path, 'w') as f:
        json.dump(remove_activations(response_data['compliant_responses']), f, indent=2)
    print(f"Compliant responses saved to: {compliant_path}")

    # Save all refusal responses
    refusal_path = os.path.join(output_dir, "refusal_responses.json")
    with open(refusal_path, 'w') as f:
        json.dump(remove_activations(response_data['refusal_responses']), f, indent=2)
    print(f"Refusal responses saved to: {refusal_path}")

    # Save filtered compliant responses
    filtered_compliant_path = os.path.join(output_dir, "filtered_compliant_responses.json")
    with open(filtered_compliant_path, 'w') as f:
        json.dump(remove_activations(response_data['filtered_compliant_responses']), f, indent=2)
    print(f"Filtered compliant responses saved to: {filtered_compliant_path}")

    # Save filtered refusal responses
    filtered_refusal_path = os.path.join(output_dir, "filtered_refusal_responses.json")
    with open(filtered_refusal_path, 'w') as f:
        json.dump(remove_activations(response_data['filtered_refusal_responses']), f, indent=2)
    print(f"Filtered refusal responses saved to: {filtered_refusal_path}")

    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"Summary Statistics")
    print(f"{'='*80}")
    print(f"Vector shape: {refusal_vector.shape}")
    print(f"Number of layers: {refusal_vector.shape[0]}")
    print(f"Hidden dimension: {refusal_vector.shape[1]}")
    print(f"\nCompliant responses: {metadata['n_compliant_responses_kept']}/{metadata['n_compliant_responses_generated']} kept ({metadata['compliant_filter_rate']*100:.1f}%)")
    print(f"Refusal responses: {metadata['n_refusal_responses_kept']}/{metadata['n_refusal_responses_generated']} kept ({metadata['refusal_filter_rate']*100:.1f}%)")
    print(f"\nOutput files saved to: {output_dir}")
    print(f"  - refusal_vector.pt")
    print(f"  - metadata.json")
    print(f"  - compliant_responses.json")
    print(f"  - refusal_responses.json")
    print(f"  - filtered_compliant_responses.json")
    print(f"  - filtered_refusal_responses.json")

    # Print per-layer statistics
    print(f"\nPer-layer L2 norms:")
    for layer_idx in range(refusal_vector.shape[0]):
        norm = refusal_vector[layer_idx].norm().item()
        print(f"  Layer {layer_idx:2d}: {norm:.4f}")

    print(f"\n{'='*80}\n")
    print("Done!")


if __name__ == "__main__":
    main()
