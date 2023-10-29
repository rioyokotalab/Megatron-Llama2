# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import re
import sys
from collections import OrderedDict

import torch

from transformers import AutoTokenizer, LlamaConfig
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint


def add_checkpointing_args(parser):
    parser.add_argument("--megatron-path", type=str, default=None, help="Base directory of Megatron repository")
    parser.add_argument(
        "--convert_checkpoint_from_megatron_to_transformers",
        action="store_true",
        help=(
            "If True, convert a Megatron checkpoint to a Transformers checkpoint. "
            "If False, convert a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the converted checkpoint.",
    )
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    return parser


def add_megatron_checkpoint_args(parser):
    parser.add_argument(
        "--target_tensor_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The tensor model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_pipeline_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The pipeline model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_data_parallel_size",
        type=int,
        default=1,
        help=(
            "The data parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_params_dtype",
        type=str,
        default="fp32",
        help=(
            "The dtype of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--make_vocab_size_divisible_by",
        type=int,
        default=128,
        help=(
            "Pad the vocab size to be divisible by this value. "
            "This is added for computational efficiency reasons. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--use_distributed_optimizer",
        action="store_true",
        help=(
            "If True, use the distributed optimizer. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    return parser


def add_transformers_checkpoint_args(parser):
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help=(
            "The name of the pre-trained tokenizer to save. "
            "If not None, the tokenizer will be saved. "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="8GB",
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`). "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )

    return parser


# megatron-lm shards mlp layers and transformer's attention layers
# so, these parameters is needed to merge, but, layernorm's weights don't need to be merged.
# megatron-lm llama model checkpoint has 'input_norm.weight' and 'post_attention_norm.weight'
# but these two should not be in tensor_parallel_params
tensor_parallel_params: list[str] = [
    # megatron-lm layers to merge across tp ranks
    "self_attention.query_key_value.weight",
    "self_attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h.weight",
    # transformers layers to split across tp ranks
    "attn.c_attn.weight",
    "attn.c_attn.bias",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
]


def recursive_print(name, val, spaces=0) -> None:
    """
    Recursively print the structure of a checkpoint. This function is taken from `convert_megatron_gpt2_checkpoint.py`

    Args:
        name (str): the name of the current tensor parameter
        val (Tuple(int)): the shape of the current tensor parameter
        spaces (int): the number of spaces to print before the output for a nested structure
    """
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def permute_qkv(qkv_w: torch.Tensor, dim: int, n_heads: int,
                n_heads_kv: int, revert: bool = False) -> torch.Tensor:

    def permute(x: torch.Tensor) -> torch.Tensor:
        if revert:
            return x.view(head_dim // 2, 2, dim).transpose(0, 1).reshape(head_dim, dim)
        return x.view(2, head_dim // 2, dim).transpose(0, 1).reshape(head_dim, dim)

    head_dim: int = dim // n_heads
    n_qs_per_kv: int = n_heads // n_heads_kv
    n_groups: int = qkv_w.size(0) // head_dim // (n_qs_per_kv + 2)
    groups: list[torch.Tensor] = torch.chunk(qkv_w, n_groups, dim=0)

    new = []
    for group in groups:
        *qs, k, v = torch.split(group, head_dim, dim=0)
        assert len(qs) == n_qs_per_kv, f"{len(qs)}, {n_qs_per_kv}"
        new += list(map(permute, qs)) + [permute(k), v]
    return torch.cat(new, dim=0)


def convert_wqkv(
    state_dict: torch.Tensor,
    layer_idx: int = 0,
    n_heads: int = 32,
    n_heads_kv: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv_w = state_dict

    n_hidden: int = qkv_w.size(1)
    hidden_dim: int = n_hidden // n_heads
    qkv_w: torch.Tensor = permute_qkv(qkv_w, n_hidden, n_heads, n_heads_kv, revert=True)

    n_qs_per_kv: int = n_heads // n_heads_kv
    n_groups: int = qkv_w.size(0) // hidden_dim // (n_qs_per_kv + 2)
    qkv_w = list(
        torch.split(qkv_w, hidden_dim, dim=0)  # type: ignore
    )

    wq, wk, wv = [], [], []
    for group in range(n_groups):
        for qs in range(n_qs_per_kv):
            wq.append(qkv_w[0])
            del qkv_w[0]  # type: ignore
        wk.append(qkv_w[0])
        del qkv_w[0]  # type: ignore
        wv.append(qkv_w[0])
        del qkv_w[0]  # type: ignore
    assert len(qkv_w) == 0

    wq = torch.concat(wq, dim=0)
    wk = torch.concat(wk, dim=0)
    wv = torch.concat(wv, dim=0)
    return wq, wk, wv


def megatron_to_transformers_fix_query_key_value_ordering(
    param: torch.Tensor,
    checkpoint_version: float,
    num_splits: int,
    num_heads: int,
    hidden_size: int
):
    """
    Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :] for compatibility with later versions
    of NVIDIA Megatron-LM. The inverse operation is performed inside Megatron-LM to read checkpoints:
    https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209 If param is the weight tensor of the
    self-attention block, the returned tensor will have to be transposed one more time to be read by HuggingFace GPT2.
    This function is taken from `convert_megatron_gpt2_checkpoint.py`

    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()

    param = param.view(*input_shape)
    return param


def merge_transformers_sharded_states(path, num_checkpoints):
    """
    Merge sharded checkpoints from transformers into a single checkpoint.

    Args:
        path (str): the path to the sharded checkpoints
        num_checkpoints (int): the number of checkpoints to merge
    """
    state_dict = {}
    for i in range(1, num_checkpoints + 1):
        checkpoint_path = os.path.join(path, f"pytorch_model-{i:05d}-of-{num_checkpoints:05d}.bin")
        current_chunk = torch.load(checkpoint_path, map_location="cpu")
        state_dict.update(current_chunk)
    return state_dict


def get_megatron_sharded_states(
    args: argparse.Namespace, tp_size: int, pp_size: int, pp_rank: int
) -> list[OrderedDict[str, torch.Tensor]]:
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.

    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts: list[OrderedDict] = []
    for i in range(tp_size):
        sub_dir_name: str = f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}"
        for checkpoint_name in ["model_optim_rng.pt", "model_rng.pt"]:
            checkpoint_path: str = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
            if os.path.isfile(checkpoint_path):
                break
        state_dict: OrderedDict = torch.load(checkpoint_path, map_location="cpu")  # type: ignore
        tp_state_dicts.append(state_dict)
    return tp_state_dicts


def get_element_from_dict_by_path(
    d: OrderedDict[str, torch.Tensor], path: str
) -> torch.Tensor | OrderedDict[str, torch.Tensor]:
    """
    Get element from dictionary by path. If element is not present, recursively add empty dictionaries.

    Args:
        d (dict): the dictionary to get the element from
        path (list): the path to the element which is delimited by "."
    """
    # path: "model.language_model.embedding.position_embeddings.weight"
    # 再帰的に 階層を深くしていく
    paths: list[str] = path.split(".")
    for key in paths:
        if key not in d:
            d[key] = {}  # type: ignore
        d = d[key]  # type: ignore
    return d  # type: ignore


def convert_ffn(
    state_dict: torch.Tensor, layer_idx: int = 0, n_dense: int = 11008
) -> tuple[torch.Tensor, torch.Tensor]:
    megatron_ffn = state_dict
    ffn_w3, ffn_w1 = megatron_ffn.split(n_dense, dim=0)
    return ffn_w1, ffn_w3


def convert_checkpoint_from_megatron_to_transformers(args: argparse.Namespace) -> None:
    """
    Convert NVIDIA Megatron-LM checkpoint to HuggingFace Transformers checkpoint. This handles Megatron checkpoints
    with different tensor parallelism and pipeline parallelism sizes. It saves the converted checkpoint into shards
    using HuggingFace Transformers checkpoint sharding functionality.

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    # Load Megatron-LM checkpoint arguments from the state dict
    sub_dirs = os.listdir(args.load_path)
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_000"]
    for sub_dir in possible_sub_dirs:
        if sub_dir in sub_dirs:
            rank0_checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir))[0]
            rank0_checkpoint_path = os.path.join(args.load_path, sub_dir, rank0_checkpoint_name)
            break
    print(f"Loading Megatron-LM checkpoint arguments from: {rank0_checkpoint_path}")  # type: ignore
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")  # type: ignore
    megatron_args: argparse.Namespace | None = state_dict.get("args", None)
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint instead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    # Create Transformers Llama config from Megatron-LM arguments
    vocab_size: int = (
        megatron_args.padded_vocab_size
        if getattr(megatron_args, "orig_vocab_size", None) is None
        else megatron_args.orig_vocab_size
    )
    print(vocab_size)

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=megatron_args.hidden_size,
        intermediate_size=megatron_args.ffn_hidden_size,
        num_hidden_layers=megatron_args.num_layers,
        num_attention_heads=megatron_args.num_attention_heads,
        num_key_value_heads=None,  # TODO: 確認
        hidden_act="silu",
        max_position_embeddings=megatron_args.max_position_embeddings,  # llama-2 では rotary-positional-embedding
        initializer_range=megatron_args.init_method_std,  # 0.02
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=megatron_args.tensor_model_parallel_size,
        tie_word_embeddings=False,
        rope_theta=10000,  # TODO: 確認
    )

    output_state_dict = {}

    tp_size: int = megatron_args.tensor_model_parallel_size
    pp_size: int = megatron_args.pipeline_model_parallel_size
    dtype = torch.bfloat16
    hidden_per_head: int = megatron_args.hidden_size // megatron_args.num_attention_heads
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, hidden_per_head, 2).float() / hidden_per_head))
    # The regex to extract layer names.
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # Convert.
    print("Converting")

    # Embeddings
    tp_state_dicts = get_megatron_sharded_states(
        args=args, tp_size=tp_size, pp_size=pp_size, pp_rank=0
    )
    print(f"DEBUG: pp=0 tp_state_dict: {tp_state_dicts}\nDEBUG: pp=1 state_dict: {get_megatron_sharded_states(args=args, tp_size=tp_size, pp_size=pp_size, pp_rank=1)}")

    # positional embedding とは異なり rotary positional embedding は checkpoint に word embedding を保存しない

    # Convert and store the word embeddings.
    word_embeddings = torch.cat(
        [
            get_element_from_dict_by_path(
                tp_state_dicts[tp_rank], "model.language_model.embedding.word_embeddings.weight"  # type: ignore
            )
            for tp_rank in range(tp_size)
        ],
        dim=0,
    )
    assert len(word_embeddings) == vocab_size
    # word_embeddings.shape: torch.Size([32000, 4096] (default Llama Tokenizer)
    word_embeddings = word_embeddings.to(dtype)
    output_state_dict["model.embed_tokens.weight"] = word_embeddings

    # Transformer Layers
    print("Converting transformer layers")
    # The number of heads.
    heads: int = config.num_attention_heads
    n_heads_kv = getattr(args, "num_attention_heads_kv", heads)
    num_layers: int = config.num_hidden_layers // pp_size

    for pp_rank in range(pp_size):
        if pp_size > 0:
            print(f"Converting pipeline parallel rank {pp_rank}")
            tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, pp_rank)

        # The transformer.
        path: str = "model.language_model.encoder"
        # Extract the layers.
        # OrderedDict[str, torch.Tensor] から key, value を取り出す
        for key, val in get_element_from_dict_by_path(tp_state_dicts[0], path).items():  # type: ignore
            # key: layers.0.self_attention.query_key_value.weight (exp)
            # val: tensor([[-0.0060, -0.0145, -0.0021,  ... ], ... ])

            # Match the name.
            m = layer_re.match(key)
            # Stop if that's not a layer
            if m is None:
                break

            # example
            # m.group(1): 0 (layer.idx)
            # m.group(2): input_norm
            # m.group(3): weight

            # The index of the layer.
            # pipeline parallel で layerが分けられているので、それを考慮する
            layer_idx: int = int(m.group(1)) + pp_rank * num_layers
            # The name of the operation.
            op_name: str = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias: str = m.group(3)

            # The name of the layer.
            layer_name: str = f"model.layers.{layer_idx}"

            if op_name + "." + weight_or_bias not in tensor_parallel_params:
                params: torch.Tensor = val.to(dtype)  # type: ignore
                print(f"DEBUG: key: {key} is not sharded by tensor parallel")
            else:
                dim = 1 if op_name in ["self_attention.dense", "mlp.dense_4h_to_h"] else 0
                params: torch.Tensor = torch.cat(
                    [val]
                    + [
                        get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]  # type: ignore
                        for tp_rank in range(1, tp_size)
                    ],
                    dim=dim,
                ).to(dtype)

            # For layernorm(s), simply store the layer norm.
            if op_name.endswith("norm"):
                # layers.idx.input_norm.weight (megatron) -> layers.idx.input_layernorm.weight (hf)
                # layers.idx.post_attention_norm.weight (megatron) -> layers.idx.post_attention_layernorm.weight (hf)
                ln_name = "input_layernorm" if op_name.startswith("input") else "post_attention_layernorm"
                # layer_name: model.layers.0
                # ln_name: input_layernorm
                # weight_or_bias: weight
                output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = params
                # rotary positional embeddings
                output_state_dict[f"{layer_name}.self_attn.rotary_emb.inv_freq"] = inv_freq.to(dtype)

            # Transpose the QKV matrix.
            elif (op_name == "self_attention.query_key_value") and weight_or_bias == "weight":
                # query_weight, key_wight, value_wight
                wq, wk, wv = convert_wqkv(
                    state_dict=params,
                    layer_idx=layer_idx,
                    n_heads=heads,
                    n_heads_kv=n_heads_kv
                )

                output_state_dict[f"{layer_name}.self_attn.q_proj.weight"] = wq
                output_state_dict[f"{layer_name}.self_attn.k_proj.weight"] = wk
                output_state_dict[f"{layer_name}.self_attn.v_proj.weight"] = wv

            elif (op_name == "self_attention.dense") and weight_or_bias == "weight":
                output_state_dict[f"{layer_name}.self_attn.o_proj.weight"] = params

            # Transpose the weights.
            elif weight_or_bias == "weight":
                # model.layers.idx.mlp.gate_proj.weight
                # model.layers.idx.mlp.up_proj.weight
                # model.layers.idx.mlp.down_proj.weight
                if op_name == "mlp.dense_h_to_4h":
                    ffn_w1, ffn_w3 = convert_ffn(
                        state_dict=params,
                        layer_idx=layer_idx,
                        n_dense=megatron_args.ffn_hidden_size,
                    )
                    output_state_dict[f"{layer_name}.mlp.gate_proj.weight"] = ffn_w1
                    output_state_dict[f"{layer_name}.mlp.up_proj.weight"] = ffn_w3
                elif op_name == "mlp.dense_4h_to_h":
                    output_state_dict[f"{layer_name}.mlp.down_proj.weight"] = params
                else:
                    print(f"DEBUG: unsupported type: op_name: {op_name}, key: {key}")

            else:
                print(f"DEBUG: unsupported type: op_name: {op_name}, key: {key}")

        # final layer
        if pp_rank == pp_size - 1:
            print(f"Converting final layer norm: pp rank = {pp_rank}")
            tp_state_dicts = get_megatron_sharded_states(
                args=args, tp_size=tp_size, pp_size=pp_size, pp_rank=pp_size - 1
            )
            encoder_params = get_element_from_dict_by_path(
                tp_state_dicts[0],
                path="model.language_model.encoder"
            )
            final_norm_weight = encoder_params["final_norm.weight"]  # type: ignore
            output_layer_weight = get_element_from_dict_by_path(
                tp_state_dicts[0],
                path="model.language_model.output_layer.weight"
            )
            print(f"DEBUG: before version shape: {output_layer_weight.shape}")  # type: ignore
            lm_head_weights: torch.Tensor = torch.cat(
                [
                    get_element_from_dict_by_path(
                        tp_state_dicts[tp_rank], "model.language_model.output_layer.weight"  # type: ignore
                    )
                    for tp_rank in range(tp_size)
                ],
                dim=0,
            )
            print(f"DEBUG: final norm weight: {final_norm_weight}, shape: {final_norm_weight.shape}")
            print(f"DEBUG: lm head weight: {lm_head_weights}, shape: {lm_head_weights.shape}")

            output_state_dict["model.norm.weight"] = final_norm_weight.to(dtype)  # type: ignore
            output_state_dict["lm_head.weight"] = lm_head_weights.to(dtype)  # type: ignore

    if config.num_hidden_layers != (layer_idx + 1):  # type: ignore
        raise ValueError(f"Expected {config.num_hidden_layers} layers but found {layer_idx + 1}")  # type: ignore

    # It should be done!
    print("Conversion from Megatron-LM to Transformers is done!")

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    # Add tokenizer class info to config
    # see https://github.com/huggingface/transformers/issues/13906)

    if args.tokenizer_name is None:
        tokenizer_name = "meta-llama/Llama-2-7b-chat-hf"
    else:
        tokenizer_name = args.tokenizer_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer_class = type(tokenizer).__name__
    config.tokenizer_class = tokenizer_class

    # Store the config to file.
    print("Saving config")
    config.save_pretrained(args.save_path)

    # Save tokenizer based on args
    if args.tokenizer_name is not None:
        print(f"Adding {tokenizer_class} tokenizer files")
        tokenizer.save_pretrained(args.save_path)

    # Store the state_dict to file.
    max_shard_size = int(args.max_shard_size) if args.max_shard_size.isdigit() else args.max_shard_size
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)

    # params count
    param_count: int = 0
    for k, v in output_state_dict.items():
        param_count += v.numel()
    print(f"total param count: {param_count}, total size: {param_count * 2}")

    # Save the model
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(args.save_path, shard_file))

    if index is None:
        print(f"Model weights saved in {os.path.join(args.save_path, WEIGHTS_NAME)}")
    else:
        save_index_file = os.path.join(args.save_path, WEIGHTS_INDEX_NAME)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        print(
            f"The model is bigger than the maximum size per checkpoint ({args.max_shard_size}) and is going to be "
            f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser = add_checkpointing_args(parser)
    parser = add_megatron_checkpoint_args(parser)
    parser = add_transformers_checkpoint_args(parser)
    args = parser.parse_args()
    if args.convert_checkpoint_from_megatron_to_transformers:
        if args.megatron_path:
            sys.path.append(args.megatron_path)
        convert_checkpoint_from_megatron_to_transformers(args)
    else:
        print("transformers to megatron is not supported")


if __name__ == "__main__":
    main()
