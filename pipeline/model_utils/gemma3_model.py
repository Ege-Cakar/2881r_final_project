
import torch
import functools

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from jaxtyping import Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Gemma 3 chat template is the same as Gemma 2
# - Official Gemma documentation: https://ai.google.dev/gemma/docs/formatting

GEMMA3_CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

# Gemma 3 uses a different tokenizer with different token IDs
GEMMA3_REFUSAL_TOKS = [236777] # ['I'] - different from Gemma 2's 235285

def format_instruction_gemma3_chat(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True,
):
    if system is not None:
        raise ValueError("System prompts are not supported for Gemma models.")
    else:
        formatted_instruction = GEMMA3_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()
    
    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_gemma3_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_gemma3_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_gemma3_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

def orthogonalize_gemma3_weights(model: AutoTokenizer, direction: Float[Tensor, "d_model"]):
    # Handle both text-only (Gemma3ForCausalLM) and multimodal (Gemma3ForConditionalGeneration)
    if hasattr(model, 'language_model'):
        # Multimodal: model.language_model
        lang_model = model.language_model
    else:
        # Text-only: model.model
        lang_model = model.model

    lang_model.embed_tokens.weight.data = get_orthogonalized_matrix(lang_model.embed_tokens.weight.data, direction)

    for block in lang_model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T

def act_add_gemma3_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    # Handle both text-only (Gemma3ForCausalLM) and multimodal (Gemma3ForConditionalGeneration)
    if hasattr(model, 'language_model'):
        # Multimodal: model.language_model.layers
        layers = model.language_model.layers
    else:
        # Text-only: model.model.layers
        layers = model.model.layers

    dtype = layers[layer-1].mlp.down_proj.weight.dtype
    device = layers[layer-1].mlp.down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    layers[layer-1].mlp.down_proj.bias = torch.nn.Parameter(bias)


class Gemma3Model(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="cuda",
        ).eval()

        model.requires_grad_(False) 

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = 'left'

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_gemma3_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        return self.tokenizer.encode(GEMMA3_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        return GEMMA3_REFUSAL_TOKS

    def _get_model_block_modules(self):
        # Handle both text-only (Gemma3ForCausalLM) and multimodal (Gemma3ForConditionalGeneration)
        if hasattr(self.model, 'language_model'):
            # Multimodal: model.language_model.layers
            return self.model.language_model.layers
        else:
            # Text-only: model.model.layers
            return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_gemma3_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_gemma3_weights, direction=direction, coeff=coeff, layer=layer)

    def get_kl_threshold(self):
        """Gemma 3 has larger activation norms, requiring a higher KL threshold."""
        return 15.0
